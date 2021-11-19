param(
    $ResourceGroupName,
    $WorkspaceName
)

$SubscriptionId = (Get-AzContext).Subscription.Id

$ConfigBody = @"
{
    "subscription_id": "$SubscriptionId",
    "resource_group": "$ResourceGroupName",
    "workspace_name": "$WorkspaceName"
}
"@


$Folder = Get-Item ".azureml/" 
if(!$Folder){
    New-Item -Name .azureml -ItemType Directory
}

New-Item -Path .azureml/config.json -Value $ConfigBody