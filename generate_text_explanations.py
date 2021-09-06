import numpy as np

def GenerateTextExplanations(explanation, dataset):

    # creating text explanation based on changed features
    input = explanation['x_cfs_highlight'].iloc[0]
    cf = explanation['x_cfs_highlight'].iloc[1]
    idx_changed = np.where(cf != '_')[0]
    text_explanation = 'IF '
    counter = 0
    for idx in idx_changed:
        feature = str(cf.index[idx])
        value = str(cf.values[idx])
        if counter <= 0:
            change = '\''+ feature +' was ' + value + '\''
        else:
            change = ' AND ' + '\'' + feature + ' was ' + value + '\''
        text_explanation += change
        counter += 1

    # numeric label
    input_class= explanation['x_cfs_eval'].iloc[0]['Class']
    cf_class = explanation['x_cfs_eval'].iloc[1]['Class']

    # original label
    input_class = dataset['labels'][input_class]
    cf_class = dataset['labels'][cf_class]

    # create text explanation template w.r.t different data sets
    if dataset['name'] == 'adult':
        input_class = 'RECEIVING' + str(input_class) + ' SALARY'
        cf_class = 'RECEIVING' + str(cf_class) + ' SALARY'

    elif dataset['name'] == 'compas-scores-two-years':
        input_class = str(input_class) + ' RECIDIVISM RISK'
        cf_class = str(cf_class) + ' RECIDIVISM RISK'

    elif dataset['name'] == 'credit-card-default':
        labels = ['No', 'Yes']
        input_class = labels[input_class] + ' DEFAULT'
        cf_class = labels[cf_class] + ' DEFAULT'

    elif dataset['name'] == 'heloc':
        input_class = str(input_class) + ' RISK'
        cf_class = str(cf_class) + ' RISK'

    elif dataset['name'] == 'heart-disease':
        input_class = 'TYPE '+ str(input_class) + ' DIAGNOSIS'
        cf_class = 'TYPE '+ str(cf_class) + ' DIAGNOSIS'


    text_explanation += ', THE PERSON WOULD BE CLASSIFIED AS ' +  '\'' +  cf_class  + '\'' + \
                        ' RATHER THAN ' +  '\'' + input_class + '\'' +'!'

    return input, text_explanation

