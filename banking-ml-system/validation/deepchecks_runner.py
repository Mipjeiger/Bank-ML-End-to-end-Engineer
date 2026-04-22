from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import full_suite
import os
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
REPORT_DIR = BASE_DIR / "reports" / "deepchecks"

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    html_path = f"{REPORT_DIR}/report_{timestamp}.html"

    # Save the reports
    result.save_as_html(html_path)

    return html_path