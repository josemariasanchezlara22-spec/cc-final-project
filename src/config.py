PROJECT_ID = "am-up-01"
REGION = "us-central1"
BUCKET_NAME = "am-up-01-credit-risk"

RAW_DATA_PATH = "data/raw/loan.csv"
PROCESSED_DATA_PATH = "data/processed/"

MODEL_NAME = "xgboost_credit_risk"
MODEL_VERSION = "xgb_optuna_v2"

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET_COLUMN = "target"

TARGET_STATUS = [
    "Fully Paid",
    "Charged Off"
]

CANDIDATE_FEATURES = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "application_type",
    "purpose",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "tax_liens"
]