from prepare_datasets import PrepareBreastCancer , PrepareCreditCardDefault, PrepareAdult

breast_cancer = PrepareBreastCancer('datasets/', 'breast-cancer.csv')
credit_card_default = PrepareCreditCardDefault('datasets/', 'default-of-credit-card-clients.csv')
adult = PrepareAdult('datasets/', 'adult.csv')
