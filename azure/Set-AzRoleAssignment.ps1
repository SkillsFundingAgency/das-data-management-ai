[CmdletBinding()]
### TODO: Get the variables passed in from the template outputs
$SubscriptionId = (Get-AzContext).Subscription.Id 
#$resourceGroupName = "das-test-datamgmtai-rg"
#$StorageAccountName = "dastestdatamgmtaistr"

### TODO: Do this better so its not duplicated script for each role assignment
# Get the MI IDs
$mlid = (Get-AzADServicePrincipal -DisplayName das-test-datamgmtai-ml).id
$mlclusterid = (Get-AzADServicePrincipal -DisplayName das-test-datamgmtai-ml/computes/cpucluster).id

# Check if role assignment exists and create if needed
$RoleAssignments = Get-AzRoleAssignment -Scope /subscriptions/$SubscriptionId/resourcegroups/$resourceGroupName/providers/Microsoft.Storage/storageAccounts/$StorageAccountName | Where-Object { $_.RoleDefinitionName -eq "Storage Blob Data Contributor" -and $_.ObjectId -eq $mlid} 

if ($RoleAssignments.Length -eq 0) {
    New-AzRoleAssignment -ObjectId $mlid`
        -RoleDefinitionName "Storage Blob Data Contributor"`
        -Scope /subscriptions/$SubscriptionId/resourcegroups/$resourceGroupName/providers/Microsoft.Storage/storageAccounts/$StorageAccountName
}

# Do the same for the cluster principal
$RoleAssignments = Get-AzRoleAssignment -Scope /subscriptions/$SubscriptionId/resourcegroups/$resourceGroupName/providers/Microsoft.Storage/storageAccounts/$StorageAccountName | Where-Object { $_.RoleDefinitionName -eq "Storage Blob Data Contributor" -and $_.ObjectId -eq $mlclusterid} 

if ($RoleAssignments.Length -eq 0) {
    New-AzRoleAssignment -ObjectId $mlclusterid`
        -RoleDefinitionName "Storage Blob Data Contributor"`
        -Scope /subscriptions/$SubscriptionId/resourcegroups/$resourceGroupName/providers/Microsoft.Storage/storageAccounts/$StorageAccountName
}