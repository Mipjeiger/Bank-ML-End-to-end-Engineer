def preprocess_input(data: dict):
    """
    Preprocess the input data for prediction.
    This features about encoding categorical variables to ensure the data not drifted."""

    # Encoding mappings
    geography_map = {
        "France": 0,
        "Germany": 1,
        "Spain": 2
    }

    gender_map = {
        "Male": 1,
        "Female": 2
    }

    card_type_map = {
        "SILVER": 0,
        "GOLD": 1,
        "PLATINUM": 2
    }

    # Convert input
    processed = [
        data["CreditScore"],
        geography_map[data["Geography"]],
        gender_map[data["Gender"]],
        data["Age"],
        data["Tenure"],
        data["Balance"],
        data["NumOfProducts"],
        int(data["HasCrCard"]),
        int(data["IsActiveMember"]),
        data["EstimatedSalary"],
        int(data["Exited"]),
        data["Complain"],
        data["SatisfactionScore"],
        card_type_map[data["CardType"]],
        data["PointEarned"],
        data["RiskScore"],
        data["BalancePerProduct"],
        data["AgeRisk"],
        data["HighValueCustomer"],
        data["LowCreditRisk"],
        data["ComplainFlag"],
        data["LowSatisfaction"]
    ]
    print(f"Total features after preprocessing: {len(processed)}")

    return processed