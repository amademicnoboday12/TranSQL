insert or replace into freq_each_token select * from (
    with fet as (
        select getvariable('last_token_id') as token_id, list_transform(list_transform(range(64),x->x/64),x->getvariable('last_token_id')/pow(500000,x)) as freqs )
select token_id, list_transform(freqs,x->cos(x)) as freq_real, list_transform(freqs,x->sin(x)) as freq_img from fet);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 0
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 0
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 0
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 0
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=0
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 0);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=0 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=0 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=0 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 1
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 1
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 1
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 1
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=1
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 1);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=1 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=1 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=1 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 2
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 2
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 2
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 2
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=2
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 2);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=2 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=2 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=2 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 3
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 3
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 3
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 3
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=3
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 3);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=3 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=3 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=3 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 4
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 4
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 4
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 4
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=4
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 4);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=4 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=4 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=4 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 5
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 5
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 5
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 5
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=5
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 5);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=5 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=5 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=5 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 6
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 6
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 6
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 6
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=6
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 6);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=6 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=6 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=6 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 7
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 7
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 7
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 7
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=7
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 7);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=7 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=7 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=7 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 8
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 8
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 8
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 8
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=8
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 8);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=8 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=8 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=8 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 9
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 9
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 9
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 9
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=9
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 9);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=9 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=9 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=9 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 10
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 10
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 10
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 10
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=10
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 10);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=10 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=10 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=10 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 11
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 11
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 11
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 11
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=11
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 11);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=11 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=11 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=11 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 12
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 12
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 12
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 12
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=12
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 12);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=12 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=12 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=12 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 13
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 13
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 13
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 13
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=13
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 13);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=13 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=13 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=13 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 14
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 14
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 14
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 14
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=14
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 14);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=14 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=14 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=14 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 15
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 15
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 15
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 15
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=15
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 15);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=15 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=15 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=15 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 16
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 16
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 16
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 16
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=16
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 16);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=16 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=16 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=16 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 17
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 17
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 17
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 17
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=17
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 17);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=17 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=17 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=17 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 18
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 18
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 18
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 18
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=18
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 18);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=18 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=18 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=18 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 19
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 19
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 19
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 19
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=19
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 19);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=19 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=19 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=19 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 20
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 20
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 20
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 20
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=20
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 20);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=20 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=20 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=20 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 21
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 21
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 21
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 21
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=21
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 21);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=21 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=21 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=21 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 22
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 22
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 22
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 22
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=22
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 22);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=22 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=22 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=22 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 23
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 23
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 23
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 23
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=23
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 23);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=23 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=23 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=23 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 24
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 24
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 24
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 24
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=24
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 24);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=24 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=24 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=24 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 25
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 25
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 25
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 25
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=25
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 25);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=25 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=25 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=25 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 26
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 26
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 26
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 26
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=26
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 26);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=26 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=26 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=26 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 27
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 27
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 27
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 27
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=27
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 27);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=27 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=27 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=27 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 28
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 28
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 28
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 28
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=28
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 28);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=28 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=28 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=28 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 29
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 29
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 29
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 29
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=29
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 29);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=29 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=29 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=29 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 30
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 30
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 30
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 30
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=30
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 30);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=30 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=30 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=30 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);





insert or replace into input_embedding_view select * from(
with current_token as (select token_id,col_tile,embedding from raw_embedding_view  where token_id=getvariable('last_token_id')),
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
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token
LEFT JOIN df_sqrt ON df_sqrt.token_id = current_token.token_id
LEFT JOIN attention_norm AS df_ffn_norm ON current_token.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 31
);



insert or replace into query_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view  where token_id=getvariable('last_token_id')),
query_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WQ AS B
            join current_token as A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 31
    ),
 query_vector as (select token_id, head_id, row_idx, sum(r) as r from query_partial_results group by token_id, head_id,row_idx),
 grouped_query_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from query_vector group by token_id,head_id),
 complex_query_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_query_vec),
positional_query_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
    from complex_query_vec as A left join freq_each_token as B on A.token_id=B.token_id)
    select token_id,head_id, view_as_real(real_part,img_part)::Float[] as query_vec from positional_query_encoding_complex 
);


insert or replace into key_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
key_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WK AS B
            join current_token AS A ON A.col_tile = B.col_tile
        WHERE
            layer_id = 31
    ),
 key_vector as (select token_id, head_id, row_idx, sum(r) as r from key_partial_results group by token_id, head_id,row_idx),
 grouped_key_vec as (select token_id,head_id,list_sort(list_zip(list(row_idx), list(r))) as grouped_r from key_vector group by token_id,head_id),
 complex_key_vec as (select token_id,head_id, collect_real(grouped_r,64) as real, collect_img(grouped_r,65) as img from grouped_key_vec),
positional_key_encoding_complex as (select A.token_id,head_id, element_neg_sum(hadmard_prod(real,freq_real),hadmard_prod(img,freq_img)) as real_part
, element_sum(hadmard_prod(real,freq_img),hadmard_prod(img,freq_real)) as img_part 
from complex_key_vec as A left join freq_each_token as B on A.token_id=B.token_id)
select  token_id,head_id, view_as_real(real_part,img_part)::Float[] as key_vec from positional_key_encoding_complex
);


insert or replace into value_cache select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
value_partial_results AS (
        SELECT
            token_id,
            B.head_id,
            B.row_tile AS row_idx,
            A.col_tile AS col_idx,
            list_dot_product(embedding, chunk) AS r
        FROM
            WV AS B
            join current_token AS A  ON A.col_tile = B.col_tile
        WHERE
            layer_id = 31
    ),
 value_vector as (select token_id, head_id, row_idx, sum(r) as r 
 from value_partial_results group by token_id, head_id,row_idx)
select token_id,head_id, collect_as_array(list(row_idx),list(r)) as value_vec 
from value_vector group by token_id,head_id
);

insert or replace into updated_embedding select * from (
with current_token as (select token_id,col_tile,embedding from input_embedding_view where token_id=getvariable('last_token_id')),
current_query as (select token_id,head_id,query_vec from query_cache  where token_id=getvariable('last_token_id')),
triu AS (
    SELECT
        Q.head_id,
        Q.token_id AS q_token_id,
        K.token_id AS k_token_id,
        /* Replace dictGet(...) with a join on key_dict. */
        list_dot_product(Q.query_vec, kd.key_vec) / sqrt(128) AS value
    FROM current_query AS Q
    -- Generate all K.token_id from input_embedding_view
    CROSS JOIN (SELECT DISTINCT token_id FROM current_token) AS K
    -- Join key_dict to get the key_vec
    JOIN key_cache kd 
      ON kd.token_id = K.token_id 
     AND kd.head_id  = floor(Q.head_id / 4)
    WHERE Q.token_id >= K.token_id
),
attention_score as (select head_id,q_token_id,k_token_id, exp(value-60) as score from triu),
exp_sums as (select head_id,q_token_id,sum(score) as summation from attention_score group by head_id,q_token_id),
softmax_score as (select A.head_id,A.q_token_id,A.k_token_id,score/summation as softmax_score from attention_score as A 
left join exp_sums as B on A.head_id=B.head_id and A.q_token_id=B.q_token_id ),
partial_score AS (
    SELECT
        qk.head_id,
        qk.q_token_id,
        /* Softmax * value_vec => array of floats */
        list_transform(vd.value_vec, x->x*qk.softmax_score) AS qkv_score
    FROM softmax_score qk
    JOIN value_cache vd
      ON vd.token_id = qk.q_token_id
     AND vd.head_id  = floor(qk.head_id / 4)
),
qkv_score as (select head_id,q_token_id,sumForEach(list(qkv_score)) as qkv_score from partial_score group by head_id,q_token_id),

--number follwing is the embedding dimension
delta_value as (select B.q_token_id,A.row_tile, floor(A.row_tile/64) as groups, sum(list_dot_product(chunk,B.qkv_score)) as value from 
Wo as A left 
join qkv_score as B on A.head_id=B.head_id 
where A.layer_id=31
group by B.q_token_id,A.row_tile),
embedding_delta as (select q_token_id,groups as col_tile,collect_as_array(list(row_tile),list(value)) as delta from delta_value group by q_token_id,groups)
select df_embedding_delta.q_token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join embedding_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.q_token_id and input.col_tile=df_embedding_delta.col_tile);

insert or replace into embedding_after_norm select * from (
with current_token as (select token_id,col_tile,embedding from updated_embedding where token_id=getvariable('last_token_id')),
df_rsqrt as (SELECT
    token_id,
    1/sqrt(sum(list_sum(list_transform(embedding,x->x*x/4096)))+0.00001) as rep_sqrt
FROM current_token
GROUP BY token_id)
SELECT
    updated_embedding.token_id as token_id,
    updated_embedding.col_tile as col_tile,
    list_transform(hadmard_prod(embedding,chunk),x->(rep_sqrt*x)::Float) AS embedding
FROM current_token AS updated_embedding
LEFT JOIN df_rsqrt ON df_rsqrt.token_id = updated_embedding.token_id
LEFT JOIN ffn_norm AS df_ffn_norm ON updated_embedding.col_tile = df_ffn_norm.col_tile
WHERE df_ffn_norm.layer_id = 31);

insert or replace into raw_embedding_view select * from (
with current_token as (select token_id,col_tile,embedding from embedding_after_norm where token_id=getvariable('last_token_id')),
W1_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r1 
from W1 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=31 and token_id=getvariable('last_token_id') group by row_tile, token_id ),
W3_agg as (select token_id, B.row_tile as row_tile,sum(list_dot_product(embedding,chunk)) as r3 
from W3 as B join current_token as A  on A.col_tile=B.col_tile 
where layer_id=31 and token_id=getvariable('last_token_id') group by row_tile, token_id),
swish as (
    select A.token_id as token_id,A.row_tile as idx, floor(A.row_tile/64) as col_tile, A.r1/(1+exp(-A.r1))*B.r3::Float as swish 
from W1_agg as A join W3_agg as B on A.token_id=B.token_id  and A.row_tile=B.row_tile
),
swish_grouped as (select token_id, col_tile::INT as col_tile, collect_as_array(list(idx),list(swish)) as chunk from swish group by token_id,col_tile),
activated as (select token_id, row_tile, floor(row_tile/64::INT) as group_id, sum(list_dot_product(A.chunk,B.chunk))::Float as value from W2 as B 
left join swish_grouped as A on A.col_tile=B.col_tile where B.layer_id=31 group by token_id,row_tile),
output_delta as (select token_id, group_id as col_tile, collect_as_array(list(row_tile),list(value)) as delta from activated group by token_id,group_id)

select df_embedding_delta.token_id as token_id,input.col_tile,element_sum(input.embedding,df_embedding_delta.delta) as embedding
    from current_token as input
    join output_delta as df_embedding_delta
    on input.token_id=df_embedding_delta.token_id and input.col_tile=df_embedding_delta.col_tile
);


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