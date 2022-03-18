# _DAS Data Management AI_

<img src="https://avatars.githubusercontent.com/u/9841374?s=200&v=4" align="right" alt="UK Government logo">

[![Build Status](https://dev.azure.com/sfa-gov-uk/Digital%20Apprenticeship%20Service/_apis/build/status/das-data-management-ai?repoName=SkillsFundingAgency%2Fdas-data-management-ai&branchName=main)](https://dev.azure.com/sfa-gov-uk/Digital%20Apprenticeship%20Service/_build/latest?definitionId=2651&repoName=SkillsFundingAgency%2Fdas-data-management-ai&branchName=main)](https://dev.azure.com/sfa-gov-uk/Digital%20Apprenticeship%20Service/_build?definitionId=2651&_a=summary)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg?longCache=true&style=flat-square)](https://en.wikipedia.org/wiki/MIT_License)


## 🚀 Infrastucture Deployment 

```
The Azure infrastructure for this project requires some additional manual steps to complete the setup. The steps to complete a full deployment are as follows:

1. Deploy the Infrastructure using [![Azure DevOps]](https://dev.azure.com/sfa-gov-uk/Digital%20Apprenticeship%20Service/_build/latest?definitionId=2651&repoName=SkillsFundingAgency%2Fdas-data-management-ai&branchName=main)](https://dev.azure.com/sfa-gov-uk/Digital%20Apprenticeship%20Service/_build?definitionId=2651&_a=summary)
2. Grant the ML and ML CPU Cluster read access to the database das-[env]-datamgmt-staging-db on das-[env]-shared-sql
    ```CREATE USER [das-[env]-datamgmtai-ml/computes/cpucluster] FROM EXTERNAL PROVIDER```
    ```ALTER ROLE db_datareader ADD member [das-[env]-datamgmtai-ml/computes/cpucluster]```
    ```CREATE USER [das-[env]-datamgmtai-ml] FROM EXTERNAL PROVIDER```
    ```ALTER ROLE db_datareader ADD member [das-[env]-datamgmtai-ml]```
3. IAM
    - The Storage Account IAM is actions by the running the Set-AzRoleAssignment.ps1 script in the Azure folder. See script for example.
    - This should be replaced with a more automated solution when a design has been accepted for automating permission changes.
4. Manually add the SQL DataStore to the ML Portal with the following details. This should be added to the template in time
    - Datastore name : datamgmtdb
    - Datastore type : Azure  SQL Database
    - Server name : das-[env]-shared-sql
    - Database : das-[env]-datamgmt-staging-db
    - Save credentials with datastore for data access : Yes
    - Authentication type : Service Principal
    - Client Details : Client ID and Secret for [das-[env]-datamgmtai-ml]
```

## 🐛 Known Issues

There is discussion ongoing about the CI design for deploying infrastructure and running models. Currently the automatic run is disabled because of data issues but the Platform Team advise this is not a suitable solution because it masks issues behind 'fake' successfull deployments. 

The WoW of the data team whereby branches are used to trigger runs and deployed up the environment stack causes problems working between teams and ensuring test data is available. 

```