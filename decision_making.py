## Decision making using Pseudo-Weights


## Decision making using High Trade-Off Solutions



def ConstructCounterfactuals(x, toolbox, fronts, dataset, mapping_scale, mapping_offset, blackbox, cf_label, priority):
    ## Constructing counterfactuals
    pop = []
    evaluation = []
    for f in fronts:
        for ind in f:
            pop.append(np.asarray(ind))
            evaluation.append(np.asarray(toolbox.evaluate(ind)))
    pop = np.asarray(pop)
    evaluation = np.asarray(evaluation)
    solutions = np.rint(pop * mapping_scale + mapping_offset)

    cfs = pd.DataFrame(data=solutions, columns=dataset['feature_names']).astype(int)
    if cf_label is None:
        response = blackbox.predict(cfs)
        evaluation = np.c_[evaluation, response]
        cfs_eval = pd.DataFrame(data=evaluation, columns=['Prediction', 'Distance', 'Proximity', 'Actionable',
                                                          'Sparsity', 'Connectedness', 'Correlation', 'Response'])
    else:
        label = blackbox.predict(cfs)
        prob = blackbox.predict_proba(cfs)[:, cf_label]
        evaluation = np.c_[evaluation, label, prob]
        cfs_eval = pd.DataFrame(data=evaluation, columns=['Prediction', 'Distance', 'Proximity', 'Actionable',
                                                          'Sparsity', 'Connectedness', 'Correlation', 'Label',
                                                          'Probability'])

    ## Applying compulsory conditions
    drop_indices = cfs_eval[(cfs_eval['Prediction'] > 0) | (cfs_eval['Proximity'] == -1) |
                            (cfs_eval['Connectedness'] == 0)].index
    cfs.drop(drop_indices, inplace=True)
    cfs_eval.drop(drop_indices, inplace=True)

    ## Sorting counterfactuals based on priority
    sort_indices = cfs_eval.sort_values(by=list(priority.keys()), ascending=list(priority.values())).index
    cfs = cfs.reindex(sort_indices)
    cfs_eval = cfs_eval.reindex(sort_indices)

    ## Dropping duplicate counterfactuals
    cfs = cfs.drop_duplicates()
    cfs_eval = cfs_eval.reindex(cfs.index)

    ## Decoding features
    cfs_decoded = FeatureDecoder(cfs, dataset['discrete_features'], dataset['feature_encoder'])

    ## Predicting counterfactuals
    if cf_label is None:
        cfs_prob = None
        cfs_y = blackbox.predict(cfs)
        return cfs, cfs_decoded, cfs_y, cfs_prob, cfs_eval
    else:
        cfs_y = blackbox.predict(cfs)
        cfs_prob = blackbox.predict_proba(cfs)
        return cfs, cfs_decoded, cfs_y, cfs_prob, cfs_eval




    ## Decision making using user-defined priority
    priority = {
        'Sparsity': 1,
        'Distance': 1,
        'Correlation': 1,
        'Actionable': 1,
        'Connectedness': 0,
        }
    cfs = ConstructCounterfactuals(x, toolbox, fronts, dataset, mapping_scale, mapping_offset, blackbox, cf_label, priority)
    x_df = pd.DataFrame(data=x.reshape(1,-1), columns=dataset['feature_names']).astype(int)
    x_decoded = FeatureDecoder(x_df, dataset['discrete_features'], dataset['feature_encoder'])


    # ## n-Closest ground truth counterfactual in the training data
    # n = 5
    # dist = np.asarray([FeatureDistance(x, cf_, feature_width, discrete_indices, continuous_indices) for cf_ in gt])
    # closest_ind = np.argsort(dist)[:n]
    # for i in range(n):
    #     theta_cf = (gt[closest_ind[i]] - mapping_offset) / mapping_scale
    #     print('instnace:', list(gt[closest_ind[i]]), 'probability:', blackbox.predict(gt[closest_ind[i]].reshape(1,-1)),
    #           'cost:',CostFunction(x, theta_x, discrete_indices, continuous_indices, mapping_scale, mapping_offset,
    #           feature_range, feature_width, blackbox, probability_thresh, cf_label, cf_range, lof_model, hdbscan_model,
    #           action_operation, action_priority, corr_models, corr, theta_cf))
