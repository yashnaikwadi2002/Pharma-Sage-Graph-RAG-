from pymilvus import connections, utility

# Connect to Milvus
connections.connect(
    alias="default",
    uri="https://in03-59aa93c7bac087f.serverless.gcp-us-west1.cloud.zilliz.com",
    user="db_59aa93c7bac087f",
    password="Pw9/l%tn,3Cd>aF>",
    secure=True
)

# List all collection names
print(utility.list_collections())


# if __name__ == "__main__":
#     collection = create_collection()
#     print("Collection created:", collection.name)
#     print("Collections in Milvus:", Collection.list_collections())
