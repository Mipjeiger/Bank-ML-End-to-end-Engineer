from feast import Entity, ValueType

customer = Entity(
    name="customer_id",
    join_keys=["customer_id"],
    value_type=ValueType.STRING,
    description="Unique identifier for Customer ID"
)