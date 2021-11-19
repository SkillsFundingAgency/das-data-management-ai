from azureml.core import Workspace
import os


def main():
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    workspace_name = os.environ.get("WORKSPACE_NAME")
    resource_group = os.environ.get("RESOURCE_GROUP")
    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group
    )
    
    print("get_workspace:")
    print(ws)

if __name__ == "__main__":
    main()