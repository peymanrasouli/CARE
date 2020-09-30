

def RecoverOriginal(x_ord, cfs_ord, dataset):
    x_org = ord2org(x_ord, dataset)
    cfs_org = ord2org(cfs_ord.to_numpy(), dataset)
    x_cfs_org= pd.concat([x_org, cfs_org])
    index = pd.Series(['x'] + ['cf_' + str(i) for i in range(len(x_cfs_org) - 1)])
    x_cfs_org = x_cfs_org.set_index(index)
    pass