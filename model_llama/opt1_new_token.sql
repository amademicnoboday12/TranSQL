insert or replace into freq_each_token select * from (
    with fet as (
        select getvariable('last_token_id') as token_id, list_transform(list_transform(range(64),x->x/64),x->getvariable('last_token_id')/pow(500000,x)) as freqs )
select token_id, list_transform(freqs,x->cos(x)) as freq_real, list_transform(freqs,x->sin(x)) as freq_img from fet);





insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=0 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
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
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
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
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
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
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_0_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_0_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_0_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_0_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=0
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 0;



with raw_token as materialized (select token_id, col_tile, embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_0_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_0_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_0_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_0_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_0_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_0_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_0_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_0_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_0_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_0_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_0_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_0_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_0_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_0_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_0_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_0_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=0 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=1 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
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
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
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
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
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
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_1_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_1_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_1_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_1_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=1
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 1;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_1_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_1_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_1_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_1_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_1_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_1_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_1_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_1_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_1_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_1_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_1_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_1_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_1_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_1_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_1_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_1_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=1 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=2 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_2_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_2_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_2_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_2_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_2_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_2_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_2_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_2_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=2
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 2;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_2_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_2_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_2_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_2_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_2_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_2_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_2_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_2_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_2_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_2_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_2_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_2_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_2_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_2_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_2_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_2_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=2 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=3 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_3_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_3_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_3_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_3_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_3_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_3_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_3_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_3_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=3
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 3;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_3_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_3_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_3_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_3_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_3_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_3_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_3_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_3_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_3_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_3_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_3_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_3_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_3_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_3_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_3_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_3_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=3 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=4 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_4_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_4_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_4_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_4_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_4_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_4_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_4_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_4_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=4
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 4;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_4_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_4_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_4_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_4_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_4_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_4_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_4_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_4_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_4_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_4_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_4_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_4_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_4_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_4_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_4_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_4_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=4 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=5 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_5_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_5_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_5_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_5_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_5_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_5_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_5_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_5_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=5
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 5;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_5_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_5_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_5_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_5_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_5_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_5_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_5_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_5_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_5_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_5_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_5_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_5_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_5_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_5_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_5_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_5_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=5 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=6 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_6_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_6_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_6_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_6_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_6_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_6_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_6_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_6_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=6
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 6;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_6_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_6_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_6_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_6_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_6_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_6_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_6_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_6_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_6_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_6_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_6_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_6_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_6_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_6_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_6_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_6_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=6 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=7 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_7_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_7_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_7_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_7_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_7_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_7_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_7_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_7_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=7
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 7;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_7_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_7_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_7_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_7_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_7_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_7_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_7_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_7_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_7_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_7_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_7_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_7_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_7_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_7_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_7_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_7_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=7 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=8 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_8_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_8_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_8_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_8_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_8_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_8_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_8_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_8_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=8
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 8;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_8_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_8_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_8_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_8_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_8_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_8_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_8_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_8_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_8_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_8_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_8_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_8_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_8_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_8_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_8_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_8_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=8 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=9 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_9_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_9_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_9_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_9_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_9_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_9_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_9_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_9_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=9
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 9;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_9_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_9_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_9_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_9_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_9_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_9_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_9_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_9_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_9_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_9_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_9_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_9_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_9_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_9_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_9_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_9_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=9 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=10 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_10_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_10_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_10_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_10_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_10_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_10_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_10_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_10_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=10
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 10;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_10_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_10_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_10_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_10_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_10_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_10_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_10_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_10_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_10_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_10_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_10_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_10_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_10_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_10_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_10_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_10_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=10 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=11 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_11_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_11_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_11_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_11_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_11_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_11_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_11_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_11_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=11
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 11;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_11_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_11_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_11_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_11_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_11_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_11_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_11_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_11_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_11_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_11_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_11_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_11_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_11_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_11_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_11_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_11_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=11 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=12 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_12_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_12_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_12_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_12_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_12_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_12_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_12_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_12_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=12
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 12;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_12_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_12_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_12_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_12_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_12_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_12_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_12_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_12_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_12_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_12_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_12_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_12_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_12_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_12_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_12_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_12_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=12 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=13 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_13_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_13_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_13_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_13_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_13_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_13_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_13_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_13_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=13
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 13;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_13_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_13_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_13_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_13_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_13_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_13_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_13_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_13_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_13_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_13_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_13_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_13_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_13_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_13_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_13_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_13_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=13 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=14 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_14_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_14_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_14_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_14_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_14_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_14_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_14_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_14_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=14
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 14;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_14_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_14_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_14_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_14_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_14_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_14_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_14_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_14_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_14_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_14_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_14_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_14_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_14_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_14_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_14_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_14_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=14 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=15 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_15_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_15_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_15_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_15_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_15_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_15_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_15_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_15_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=15
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 15;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_15_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_15_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_15_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_15_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_15_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_15_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_15_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_15_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_15_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_15_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_15_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_15_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_15_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_15_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_15_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_15_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=15 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=16 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_16_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_16_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_16_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_16_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_16_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_16_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_16_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_16_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=16
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 16;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_16_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_16_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_16_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_16_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_16_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_16_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_16_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_16_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_16_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_16_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_16_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_16_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_16_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_16_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_16_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_16_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=16 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=17 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_17_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_17_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_17_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_17_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_17_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_17_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_17_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_17_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=17
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 17;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_17_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_17_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_17_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_17_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_17_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_17_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_17_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_17_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_17_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_17_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_17_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_17_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_17_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_17_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_17_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_17_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=17 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=18 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_18_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_18_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_18_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_18_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_18_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_18_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_18_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_18_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=18
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 18;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_18_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_18_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_18_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_18_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_18_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_18_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_18_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_18_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_18_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_18_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_18_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_18_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_18_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_18_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_18_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_18_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=18 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=19 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_19_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_19_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_19_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_19_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_19_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_19_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_19_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_19_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=19
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 19;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_19_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_19_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_19_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_19_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_19_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_19_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_19_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_19_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_19_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_19_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_19_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_19_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_19_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_19_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_19_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_19_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=19 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=20 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_20_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_20_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_20_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_20_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_20_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_20_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_20_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_20_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=20
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 20;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_20_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_20_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_20_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_20_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_20_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_20_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_20_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_20_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_20_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_20_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_20_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_20_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_20_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_20_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_20_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_20_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=20 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=21 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_21_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_21_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_21_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_21_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_21_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_21_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_21_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_21_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=21
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 21;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_21_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_21_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_21_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_21_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_21_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_21_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_21_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_21_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_21_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_21_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_21_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_21_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_21_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_21_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_21_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_21_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=21 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=22 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_22_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_22_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_22_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_22_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_22_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_22_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_22_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_22_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=22
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 22;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_22_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_22_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_22_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_22_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_22_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_22_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_22_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_22_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_22_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_22_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_22_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_22_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_22_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_22_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_22_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_22_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=22 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=23 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_23_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_23_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_23_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_23_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_23_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_23_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_23_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_23_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=23
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 23;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_23_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_23_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_23_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_23_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_23_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_23_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_23_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_23_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_23_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_23_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_23_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_23_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_23_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_23_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_23_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_23_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=23 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=24 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_24_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_24_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_24_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_24_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_24_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_24_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_24_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_24_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=24
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 24;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_24_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_24_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_24_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_24_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_24_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_24_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_24_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_24_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_24_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_24_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_24_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_24_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_24_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_24_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_24_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_24_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=24 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=25 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_25_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_25_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_25_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_25_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_25_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_25_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_25_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_25_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=25
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 25;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_25_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_25_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_25_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_25_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_25_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_25_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_25_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_25_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_25_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_25_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_25_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_25_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_25_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_25_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_25_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_25_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=25 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=26 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_26_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_26_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_26_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_26_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_26_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_26_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_26_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_26_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=26
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 26;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_26_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_26_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_26_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_26_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_26_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_26_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_26_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_26_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_26_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_26_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_26_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_26_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_26_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_26_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_26_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_26_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=26 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=27 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_27_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_27_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_27_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_27_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_27_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_27_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_27_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_27_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=27
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 27;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_27_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_27_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_27_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_27_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_27_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_27_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_27_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_27_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_27_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_27_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_27_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_27_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_27_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_27_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_27_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_27_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=27 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=28 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_28_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_28_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_28_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_28_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_28_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_28_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_28_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_28_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=28
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 28;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_28_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_28_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_28_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_28_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_28_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_28_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_28_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_28_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_28_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_28_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_28_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_28_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_28_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_28_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_28_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_28_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=28 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=29 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_29_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_29_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_29_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_29_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_29_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_29_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_29_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_29_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=29
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 29;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_29_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_29_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_29_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_29_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_29_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_29_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_29_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_29_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_29_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_29_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_29_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_29_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_29_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_29_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_29_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_29_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=29 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=30 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_30_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_30_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_30_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_30_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_30_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_30_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_30_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_30_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=30
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 30;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_30_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_30_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_30_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_30_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_30_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_30_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_30_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_30_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_30_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_30_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_30_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_30_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_30_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_30_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_30_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_30_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=30 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;



insert or replace into input_embedding_view select * from(
with current_token as materialized (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id') order by col_tile),
df_sqrt as (
        select
        token_id,
    1/sqrt(sum(list_aggregate(list_transform(embedding,x->x*x/4096),'sum'))+0.00001) as rep_sqrt
        FROM current_token
        GROUP BY token_id
    )
    SELECT
    current_token.token_id as token_id,
    current_token.col_tile as col_tile,
    hadmard_prod_scalar(embedding,B.chunk,rep_sqrt) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
positional join (select chunk from attention_norm where layer_id=31 order by col_tile) as B
);


insert or replace into qkv_cache select * from (
with input_embedding_view_pv as materialized (pivot (select * from input_embedding_view where token_id=getvariable('last_token_id')) on col_tile using first(embedding)),
r1 as (
SELECT 
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
FROM WQKV_31_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r2 as (
SELECT 
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum2
FROM WQKV_31_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r3 as (
SELECT 
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3
FROM WQKV_31_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r4 as (
SELECT 
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum4
FROM WQKV_31_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r5 as (
SELECT 
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum5
FROM WQKV_31_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r6 as (
SELECT 
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum6
FROM WQKV_31_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r7 as (
SELECT 
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum7
FROM WQKV_31_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
r8 as (
SELECT 
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum8
FROM WQKV_31_pv AS B 
CROSS JOIN input_embedding_view_pv AS A
ORDER BY B.type,B.head_id ,B.row_tile),
qkv_vector as (select getvariable('last_token_id') as token_id, r1.type,r1.head_id,r1.row_tile as row_idx, (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)::Float as r from r1 positional join r2 positional join r3 positional join r4 
positional join r5 positional join r6 positional join r7 positional join r8),
grouped_qkv_vec as materialized (select type,token_id,head_id,array_agg(r order by row_idx) as grouped_r from qkv_vector group by token_id,head_id,type),
 complex_qk_vec as (select type,token_id,head_id, grouped_r[:64] as real, grouped_r[65:] as img from grouped_qkv_vec where type=0 or type=1),
positional_qk_encoding_complex as (select type,A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_qk_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select token_id,type,head_id, (real_part||img_part)::Float[] as vec from positional_qk_encoding_complex 
union all select token_id,type,head_id,grouped_r::Float[] as vec from grouped_qkv_vec where type=2
);


with query_cache_tmp as (select head_id,token_id,vec as query_vec from qkv_cache where type=0 and token_id=getvariable('last_token_id')),
key_cache_tmp as (select head_id,token_id,vec as key_vec from qkv_cache where type=1),
value_cache_tmp as (select head_id,token_id,vec as value_vec from qkv_cache where type=2),
attention_score AS materialized (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        kd.token_id AS k_token_id,
        exp(list_dot_product(Q.query_vec, kd.key_vec)/ sqrt(128)-60) AS score
    FROM query_cache_tmp AS Q
    JOIN key_cache_tmp kd 
     ON kd.head_id  = floor(Q.head_id / 4)
),
exp_sums as (select head_id,q_token_id,sum(score)::Float as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,(score/summation)::Float as softmax_score from attention_score as A left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->(x*qk.softmax_score)::Float) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache_tmp vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=31
group by B.q_token_id,A.row_tile),
current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id') order by col_tile),
embedding_delta as (select q_token_id,groups as col_tile,array_agg(value order by row_tile)::Float[] as delta from delta_value group by q_token_id,groups order by groups),
updated_embedding_local as (select token_id, current_token.col_tile, element_sum(embedding,delta) as embedding from current_token positional join embedding_delta),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM updated_embedding_local
GROUP BY token_id)
insert or replace into embedding_after_norm  SELECT
    updated_embedding_local.token_id as token_id,
    updated_embedding_local.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM updated_embedding_local
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding_local.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding_local.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 31;



with raw_token as materialized (select token_id, col_tile, embedding from updated_embedding where token_id=getvariable('last_token_id')),
current_token as materialized (pivot raw_token on col_tile using first(embedding)),
l1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum1_1,

FROM W1_31_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum1_2,

FROM W1_31_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum1_3,

FROM W1_31_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum1_4,

FROM W1_31_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum1_5,

FROM W1_31_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum1_6,

FROM W1_31_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum1_7,

FROM W1_31_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
l8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum1_8,
FROM W1_31_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r1 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."0",  B."0"),
    list_dot_product(A."1",  B."1"),
    list_dot_product(A."2",  B."2"),
    list_dot_product(A."3",  B."3"),
    list_dot_product(A."4",  B."4"),
    list_dot_product(A."5",  B."5"),
    list_dot_product(A."6",  B."6"),
    list_dot_product(A."7",  B."7")]) as sum3_1,

FROM W3_31_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r2 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."8",  B."8"),
    list_dot_product(A."9",  B."9"),
    list_dot_product(A."10", B."10"),
    list_dot_product(A."11", B."11"),
    list_dot_product(A."12", B."12"),
    list_dot_product(A."13", B."13"),
    list_dot_product(A."14", B."14"),
    list_dot_product(A."15", B."15")]) as sum3_2,

FROM W3_31_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r3 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."16", B."16"),
    list_dot_product(A."17", B."17"),
    list_dot_product(A."18", B."18"),
    list_dot_product(A."19", B."19"),
    list_dot_product(A."20", B."20"),
    list_dot_product(A."21", B."21"),
    list_dot_product(A."22", B."22"),
    list_dot_product(A."23", B."23")]) as sum3_3,

FROM W3_31_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r4 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."24", B."24"),
    list_dot_product(A."25", B."25"),
    list_dot_product(A."26", B."26"),
    list_dot_product(A."27", B."27"),
    list_dot_product(A."28", B."28"),
    list_dot_product(A."29", B."29"),
    list_dot_product(A."30", B."30"),
    list_dot_product(A."31", B."31")]) as sum3_4,

FROM W3_31_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r5 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."32", B."32"),
    list_dot_product(A."33", B."33"),
    list_dot_product(A."34", B."34"),
    list_dot_product(A."35", B."35"),
    list_dot_product(A."36", B."36"),
    list_dot_product(A."37", B."37"),
    list_dot_product(A."38", B."38"),
    list_dot_product(A."39", B."39")]) as sum3_5,

FROM W3_31_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r6 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."40", B."40"),
    list_dot_product(A."41", B."41"),
    list_dot_product(A."42", B."42"),
    list_dot_product(A."43", B."43"),
    list_dot_product(A."44", B."44"),
    list_dot_product(A."45", B."45"),
    list_dot_product(A."46", B."46"),
    list_dot_product(A."47", B."47")]) as sum3_6,

FROM W3_31_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r7 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."48", B."48"),
    list_dot_product(A."49", B."49"),
    list_dot_product(A."50", B."50"),
    list_dot_product(A."51", B."51"),
    list_dot_product(A."52", B."52"),
    list_dot_product(A."53", B."53"),
    list_dot_product(A."54", B."54"),
    list_dot_product(A."55", B."55")]) as sum3_7,

FROM W3_31_pv AS B 
CROSS JOIN current_token AS A
ORDER BY A.token_id, B.row_tile),
r8 as (
SELECT 
  A.token_id,
  B.row_tile AS row_tile,
  list_sum([
    list_dot_product(A."56", B."56"),
    list_dot_product(A."57", B."57"),
    list_dot_product(A."58", B."58"),
    list_dot_product(A."59", B."59"),
    list_dot_product(A."60", B."60"),
    list_dot_product(A."61", B."61"),
    list_dot_product(A."62", B."62"),
    list_dot_product(A."63", B."63")]) as sum3_8,
FROM W3_31_pv AS B 
CROSS JOIN current_token AS A
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
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped_local as A on A.col_tile=B.col_tile where B.layer_id=31 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)
insert or replace into raw_embedding_view select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from raw_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile;
insert into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from raw_embedding_view where token_id=getvariable('last_token_id')),
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
LEFT JOIN final_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile),
output_token_encode as (select row_tile,sum(list_dot_product(embedding,chunk)) as logit 
from output_weight join output_embedding as last_token
on last_token.col_tile=output_weight.col_tile
group by row_tile order by logit desc limit 1)
select 
            getvariable('last_token_id')+1 as token_id,
            col_tile,
            embedding
            from vocabulary join output_token_encode
            on vocabulary.token_encode=output_token_encode.row_tile
);

set variable last_token_id= (select getvariable('last_token_id')+1);