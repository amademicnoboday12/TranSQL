CREATE MACRO hadmard_prod_scalar(arr1, arr2, scalar) AS (
      list_transform(list_zip(arr1,arr2),x->x[1]*x[2]*scalar));
CREATE MACRO hadmard_prod(arr1, arr2) AS (
      list_transform(list_zip(arr1,arr2),x->x[1]*x[2]));
create macro element_neg_sum(arr1,arr2) as (
      list_transform(list_zip(arr1,arr2),x->x[1]-x[2])
);
create macro element_sum(arr1,arr2) as (
      list_transform(list_zip(arr1,arr2),x->x[1]+x[2])
);

create macro element_sum_quad_avg(arr1,arr2, dim) as (
      list_transform(list_zip(arr1,arr2),x->(x[1]+x[2])**2/dim)
);
create macro view_as_real(arr1, arr2) as (
     list_concat(arr1,arr2) 
);

create macro collect_as_array(idx, arr) as (
      list_transform(list_sort(list_zip(idx,arr)),x->x[2])
);

create macro collect_real(ziped_arr,mid_pos) as (
      list_transform(ziped_arr[:mid_pos],x->x[2])
);
create macro collect_img(ziped_arr,mid_pos) as (
      list_transform(ziped_arr[mid_pos:],x->x[2])
);

create macro sumForEach(arr) as (
      list_reduce(arr, (acc, row)-> list_transform(acc, (acc_val, i)->acc_val+row[i]))
);