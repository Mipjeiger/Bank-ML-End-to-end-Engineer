from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import full_suite
import os
from datetime import datetime

REPORT_DIR = "reports/deepchecks"

# Create function for running deepchecks suite
def run_full_validation(train_df, test_df, model, label="target"):
    train_ds = Dataset(train_df, label=label)
    test_ds = Dataset(test_df, label=label)

    suite = full_suite()
    
    result = suite.run(
        train_dataset=train_ds,
        test_dataset=test_ds,
        model=model
    )

    # Save the report
    os.makedirs(REPORT_DIR, exist_ok=True)

    file_path = os.path.join(
        REPORT_DIR,
        f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    result.save_as_html(file_path)

    return file_path