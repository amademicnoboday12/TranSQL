truncate sentence;
truncate freq_each_token;
truncate raw_embedding_view;
truncate input_embedding_view;
truncate qkv_cache;
truncate updated_embedding;
truncate embedding_after_norm;



insert into sentence values (0,103567),(1,49),(2,3390),(3,25),(4,3639),(5,596),(6,5076),(7,304),(8,1475),(9,11240),(10,30);

insert into freq_each_token select * from (with fet as (select token_id, list_transform(a.cis,x->x*token_id::Int32) as freqs from sentence,  (select list_transform(list_transform(range(64),x->x/64),x->1/pow(10000,x)) as cis) as a) 
select token_id, list_transform(freqs,x->cos(x)) as freq_real, list_transform(freqs,x->sin(x)) as freq_img from fet);

insert into raw_embedding_view select * from (
    SELECT
            token_id,
            col_tile,
            embedding
        FROM sentence AS input_sentence
        INNER JOIN vocabulary ON input_sentence.token_encode = vocabulary.token_encode
);


insert or replace into input_embedding_view select * from ( 
with df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/2048)))+0.00001) as rep_sqrt
FROM raw_embedding_view
GROUP BY token_id)
SELECT
    raw_embedding_view.token_id as token_id,
    raw_embedding_view.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM raw_embedding_view
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = raw_embedding_view.token_id
LEFT JOIN normalizations ON raw_embedding_view.col_tile = normalizations.col_tile
WHERE normalizations.layer_id = 0 and normalizations.type=0);

--QKV
with input_embedding_view_pv  as materialized (pivot input_embedding_view on col_tile using first(embedding)),
r1 as (
SELECT 
  A.token_id,
  B.type,
  B.head_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1
FROM WQKV_0_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY A.token_id,B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.type,
  B.head_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_0_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY A.token_id,B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.type,
  B.head_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_0_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY A.token_id,B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.type,
  B.head_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_0_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY A.token_id,B.type,B.head_id ,B.row_tile),
qkv_vector as (select r1.token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4)::Float as r from r1 positional join r2 positional join r3 positional join r4),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
insert or replace into qkv_cache select token_id,type,head_id, view_as_real(real_part,img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2

--self attn
with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128) AS value
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
      ON Q.token_id >= kd.token_id 
     AND kd.head_id  = Q.head_id
),
attention_score as materialized (select head_id,q_token_id,k_token_id, exp(value-60)::Float as score from triu),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score 
from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = qk.head_id
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
attention_o as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=0
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups)
insert or replace into updated_embedding select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from input_embedding_view as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile;

--post attn norm
with df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/2048)))+0.00001) as rep_sqrt
FROM updated_embedding
GROUP BY token_id)
insert or replace into embedding_after_norm  (SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN normalizations as post_attention_norm ON updated_embedding.col_tile = post_attention_norm.col_tile
WHERE post_attention_norm.layer_id = 0 and post_attention_norm.type=1);

--SWIGLU
with embedding_after_norm_pv as materialized  (pivot embedding_after_norm on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3")]) as sum1_1
FROM W1_0_pv AS B 
CROSS JOIN embedding_after_norm_pv AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_2
FROM W1_0_pv AS B 
CROSS JOIN embedding_after_norm_pv AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11")]) as sum1_3
FROM W1_0_pv AS B 
CROSS JOIN embedding_after_norm_pv AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_4
FROM W1_0_pv AS B 
CROSS JOIN embedding_after_norm_pv AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19")]) as sum1_5
FROM W1_0_pv AS B 
CROSS JOIN embedding_after_norm_pv AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_6
FROM W1_0_pv AS B 
CROSS JOIN embedding_after_norm_pv AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27")]) as sum1_7
FROM W1_0_pv AS B 
CROSS JOIN embedding_after_norm_pv AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_8
FROM W1_0_pv AS B 
CROSS JOIN embedding_after_norm_pv AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3")]) as sum3_1
FROM W3_0_pv AS B 
CROSS JOIN embedding_after_norm_pv AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_2
FROM W3_0_pv AS B 
CROSS JOIN embedding_after_norm_pv AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11")]) as sum3_3
FROM W3_0_pv AS B 
CROSS JOIN embedding_after_norm_pv AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_4
FROM W3_0_pv AS B 
CROSS JOIN embedding_after_norm_pv AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19")]) as sum3_5
FROM W3_0_pv AS B 
CROSS JOIN embedding_after_norm_pv AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_6
FROM W3_0_pv AS B 
CROSS JOIN embedding_after_norm_pv AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27")]) as sum3_7
FROM W3_0_pv AS B 
CROSS JOIN embedding_after_norm_pv AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_8
FROM W3_0_pv AS B 
CROSS JOIN embedding_after_norm_pv AS A
ORDER BY A.token_id, B.row_tile),
glu_left as (select l1.token_id, l1.row_tile, floor(l1.row_tile/64)::INT as col_tile,  
  (sum1_1+sum1_2+sum1_3+sum1_4+sum1_5+sum1_6+sum1_7+sum1_8)/(1+exp(-(sum1_1+sum1_2+sum1_3+sum1_4+sum1_5+sum1_6+sum1_7+sum1_8)))as glu
from l1 
positional join l2 
positional join l3 
positional join l4
positional join l5
positional join l6
positional join l7
positional join l8),
swish_activate as (select r1.token_id, r1.row_tile, floor(r1.row_tile/64)::INT as col_tile,  
  ((sum3_1+sum3_2+sum3_3+sum3_4+sum3_5+sum3_6+sum3_7+sum3_8)*glu)::Float as swish
from r1 
positional join r2 
positional join r3 
positional join r4
positional join r5
positional join r6
positional join r7
positional join r8
positional join glu_left),
swish_grouped_local as materialized (select token_id, col_tile, array_agg(swish order by row_tile) as chunk from swish_activate group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from mlp_down_proj as B 
join swish_grouped_local as A on A.col_tile=B.col_tile where B.row_tile<512 group by token_id,row_tile
union all
(select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from mlp_down_proj as B 
join swish_grouped_local as A on A.col_tile=B.col_tile where B.row_tile between 512 and 1023 group by token_id,row_tile)
union all
(select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from mlp_down_proj as B 
join swish_grouped_local as A on A.col_tile=B.col_tile where B.row_tile between 1024 and 1535 group by token_id,row_tile)
union all (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from mlp_down_proj as B 
join swish_grouped_local as A on A.col_tile=B.col_tile where B.row_tile between 1536 and 2047 group by token_id,row_tile)),
output_delta as (select token_id, group_id as col_tile, array_agg(value order by row_tile) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from embedding_after_norm as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;

--MOE layers
--attention
insert or replace into input_embedding_view select * from ( 
with df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/2048),'sum'))+0.00001) as rep_sqrt
FROM raw_embedding_view
GROUP BY token_id)
SELECT
    raw_embedding_view.token_id as token_id,
    raw_embedding_view.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM raw_embedding_view
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = raw_embedding_view.token_id
LEFT JOIN normalizations AS df_ffn_norm ON raw_embedding_view.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 1 and df_ffn_norm.type=0);

insert or replace into qkv_cache select * from (
with input_embedding_view_pv  as materialized (pivot input_embedding_view on col_tile using first(embedding)),
r1 as (
SELECT 
  A.token_id,
  B.type,
  B.head_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1
FROM WQKV_1_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY A.token_id,B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.type,
  B.head_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_1_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY A.token_id,B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.type,
  B.head_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_1_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY A.token_id,B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.type,
  B.head_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_1_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY A.token_id,B.type,B.head_id ,B.row_tile),
qkv_vector as (select r1.token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4)::Float as r from r1 positional join r2 positional join r3 positional join r4),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, view_as_real(real_part,img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);

--self attn
insert or replace into updated_embedding select * from (
with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128) AS value
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
      ON Q.token_id >= kd.token_id 
     AND kd.head_id  = Q.head_id
),
attention_score as materialized (select head_id,q_token_id,k_token_id, exp(value-60)::Float as score from triu),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score 
from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = qk.head_id
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
attention_o as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=1
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from input_embedding_view as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

--post attn norm
with df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/2048)))+0.00001) as rep_sqrt
FROM updated_embedding
GROUP BY token_id)
insert or replace into embedding_after_norm  (SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN normalizations as post_attention_norm ON updated_embedding.col_tile = post_attention_norm.col_tile
WHERE post_attention_norm.layer_id = 1 and normalizations.type=1);

--gate softmax
with gated_results as materialized (
    select token_id, B.row_tile, exp(-sum(list_dot_product(embedding,chunk)))::Float as r from embedding_after_norm as a
    join gate_weight as B on A.col_tile=B.col_tile where B.layer_id=1 group by token_id,B.row_tile),
summation as (select token_id,sum(r)::Float as exp_sum from gated_results group by token_id),
expert_list as (select A.token_id,A.row_tile as expert_id,r/exp_sum as score, row_number() over (partition by A.token_id order by score desc) as odr 
from gated_results as A left join summation as B on A.token_id=B.token_id qualify odr<=6
),
scored_embedding as materialized (select A.token_id, A.expert_id, A.score, B.col_tile, B.embedding from expert_list as A join embedding_after_norm as B on A.token_id=B.token_id),
multiplied as (select A.token_id, A.expert_id, B.row_tile, 
sum(list_dot_product(A.embedding,B."0")) as score0,  sum(list_dot_product(A.embedding,B."1")*A.score) as score1
from scored_embedding as A
join expert_gate_up_pv as B 
on A.expert_id=B.expert_id and A.col_tile=B.col_tile 
group by A.token_id, A.expert_id, B.row_tile
),
swish_activate as (select A.token_id, A.expert_id, A.row_tile, floor(A.row_tile/64)::INT as col_tile, (A.score0/(1+exp(-A.score0))*A.score1)::Float as swish 
from multiplied as A),
swish_grouped_local as (select token_id, expert_id, col_tile, array_agg(swish order by row_tile) as chunk from swish_activate group by token_id,expert_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value 
from experts_down_proj as B 
left join swish_grouped_local as A 
on A.col_tile=B.col_tile and A.expert_id=B.expert_id
where B.layer_id=1 
group by token_id,row_tile),
e7_s1 as (select B.token_id, A.row_tile, sum(list_dot_product(A."0",B.embedding)) as sum0_1, sum(list_dot_product(A."1",B.embedding)) as sum1_1  
from expert_shared_gateup_1 as A join embedding_after_norm as B
on A.col_tile=B.col_tile group by B.token_id,A.row_tile order by token_id, row_tile),
e7_s2 as (select token_id, row_tile, floor(row_tile/64)::INT as col_tile,sum0_1/(1+exp(-sum0_1))*sum1_1 as swish7 from e7_s1),
e8_s1 as (select B.token_id, A.row_tile, sum(list_dot_product(A."0",B.embedding)) as sum0_1, sum(list_dot_product(A."1",B.embedding)) as sum1_1  
from expert_shared_gateup_2 as A join embedding_after_norm as B
on A.col_tile=B.col_tile group by B.token_id,A.row_tile order by token_id, row_tile),
e8_s2 as (select sum0_1/(1+exp(-sum0_1))*sum1_1 as swish8 from e8_s1),
expert_shared_activate as (select token_id,row_tile,col_tile,swish7,swish8
from e7_s2 positional join  e8_s2),
left_m as materialized (select token_id, col_tile,
array_agg(swish7::Float order by row_tile) as chunk7,
array_agg(swish8::Float order by row_tile) as chunk8
from expert_shared_activate group by token_id,col_tile),
r7 as (
  select 
  token_id, row_tile, floor(row_tile/64)::INT as group_id, 
  sum(list_dot_product(A.chunk7,B."64")) as value
  from left_m as A join expert_down_all_pv as B on A.col_tile=B.col_tile 
  group by token_id,row_tile
),
r8 as (
  select 
  sum(list_dot_product(A.chunk8,B."65")) as value
  from left_m as A join expert_down_all_pv as B on A.col_tile=B.col_tile 
  group by token_id,row_tile
),
activated_shared as (
  select r7.token_id, r7.row_tile, r7.group_id, r7.value + r8.value as value 
  from  r7
  positional join r8
),
output_delta as (select token_id, group_id as col_tile, array_agg(value order by row_tile) as delta from activated group by token_id,group_id
union all
select token_id, group_id as col_tile, array_agg(value order by row_tile) as delta 
  from activated_shared 
  group by token_id, group_id
)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from embedding_after_norm as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;


insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from raw_embedding_view where token_id=10),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id),
output_embedding as (SELECT
    current_token.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = current_token.token_id
LEFT JOIN normalizations AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
where df_ffn_norm.type=2),
output_token_encode as (select row_tile,sum(list_dot_product(embedding,chunk)) as logit 
from output_weight join output_embedding as last_token
on last_token.col_tile=output_weight.col_tile
group by row_tile order by logit desc limit 1)
select 
            10+1 as token_id,
            col_tile,
            embedding
            from vocabulary join output_token_encode
            on vocabulary.token_encode=output_token_encode.row_tile
);

set variable last_token_id=10+1;