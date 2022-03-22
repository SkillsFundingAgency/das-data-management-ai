###TODO: This has been removed from the pipeline because the SP doesnt have user access admin role. This script needs to be run manually.

### Example 
### ./azure/Set-MLResourcePermissions.ps1 AzureMLWorkspaceName "mlworkspacename" -AzureMLClusterName "clustername" -StorageAccountName "storageaccountname"

[CmdletBinding()]
Param(
    [Parameter(Mandatory = $true)]
    [String]$AzureMLWorkspaceName,
    [Parameter(Mandatory = $true)]
    [String]$AzureMLClusterName,
    [Parameter(Mandatory = $true)]
    [String]$StorageAccountName
)

$SubscriptionId = (Get-AzContext).Subscription.Id 

### TODO: Do this better so its not duplicated script for each role assignment
# Get the MI IDs. ML needs two SPs to have IAM granted
$MLResourceObjectId = (Get-AzADServicePrincipal -DisplayName $AzureMLWorkspaceName).id
$MLClusterObjectId = (Get-AzADServicePrincipal -DisplayName "$AzureMLWorkspaceName/computes/$AzureMLClusterName").id

# Check if role assignment exists and create if needed
$StorageAccount = Get-AzResource -ResourceType 'Microsoft.Storage/storageAccounts' -Name $StorageAccountName

$MLResourceObjectId, $MLClusterObjectId | foreach {
    $RoleAssignments = Get-AzRoleAssignment -Scope $StorageAccount.ResourceId | Where-Object { $_.RoleDefinitionName -eq "Storage Blob Data Contributor" -and $_.ObjectId -eq $_ } 

    if ($RoleAssignments.Length -eq 0) {
        New-AzRoleAssignment -ObjectId $_ -RoleDefinitionName "Storage Blob Data Contributor" -Scope $StorageAccount.ResourceId
    }
}