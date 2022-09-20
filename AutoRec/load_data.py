import numpy as np
import pandas as pd

def get_validation_data(
    matrix_size: int = 3,
    users_count: int = 250,
    items_count: int = 100,
    seed: int = 2022
):
    np.random.seed(seed)
    
    users_mean = np.random.rand(matrix_size)
    items_mean = np.random.rand(matrix_size)

    #covariance matrix for distrubutions
    users_cov = np.random.rand(matrix_size, matrix_size)
    users_cov = np.dot(users_cov, users_cov.transpose())


    items_cov = np.random.rand(matrix_size, matrix_size)
    items_cov = np.dot(items_cov, items_cov.transpose())
    
    #create data frames
    users = pd.DataFrame(np.random.multivariate_normal(users_mean, users_cov, users_count) * 
                         np.random.laplace(loc = 1 + np.random.rand(), scale = np.random.rand(), size = (users_count, matrix_size)) 
                         , columns = ['user_' + str(i) for i in range(matrix_size)])
    users['user_id'] = [i for i in range(users_count)]

    items = pd.DataFrame(np.random.multivariate_normal(items_mean, items_cov, items_count) *
                         np.random.lognormal(mean = np.random.rand(), sigma = np.random.rand(), size = (items_count, matrix_size)),
                         columns = ['item_' + str(i) for i in range(matrix_size)])
    items['item_id'] = [i for i in range(items_count)]

    users['_merge_key'] = 1
    items['_merge_key'] = 1

    #merge data frames
    df_merge = pd.merge(users, items, on="_merge_key")
    df_merge = df_merge.drop(["_merge_key"], axis=1)
    
    users_matrix = np.asarray(df_merge.drop(['user_id', 'item_id'], axis = 1).iloc[:, 0:matrix_size] )
    items_matrix = np.asarray(df_merge.drop(['user_id', 'item_id'], axis = 1).iloc[:, matrix_size:])
    
    #calculation product and scalar product of vectors
    product = users_matrix * items_matrix
    scal_product = []
    for i in product:
        scal_product.append(np.sum(i))
    scal_product = np.asarray(scal_product)
    
    #calculation of vectors length
    user_len = []
    for i in users_matrix:
        user_len.append(np.sqrt(np.sum(i * i)))
    user_len = np.asarray(user_len)

    item_len = []
    for i in items_matrix:
        item_len.append(np.sqrt(np.sum(i * i)))
    item_len = np.asarray(item_len)

    #normalazed cos sim of vectors as response
    cos_sim_norm = 1 - np.arccos(scal_product/(user_len * item_len))/np.pi
    
    df_merge['rating'] = np.round(cos_sim_norm * 10) % 10 + 1
    
    
    mink_distance = []
    for i in (users_matrix - items_matrix)**3:
        mink_distance.append(np.sum(i))

    mink_distance = np.sign(mink_distance) * np.power(np.abs(mink_distance), 1/3)
    mink_distance = (mink_distance - mink_distance.min()) / (mink_distance.max() - mink_distance.min())
    mink_distance = np.round(mink_distance * 10) % 10 + 1
    real_heu = pd.DataFrame({'user_id' : df_merge.user_id, 'item_id' : df_merge.item_id, 'rating' : mink_distance})
    
    return df_merge, real_heu