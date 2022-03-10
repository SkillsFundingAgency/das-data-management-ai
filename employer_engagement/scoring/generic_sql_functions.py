from azureml.core import Workspace
from azureml.core.compute import ComputeTarget
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData, StepSequence, PublishedPipeline
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import PipelineEndpoint
import azureml.core
import os
from azureml.data.datapath import DataPath
from azureml.core import Workspace, Datastore, Dataset, ComputeTarget, Experiment, ScriptRunConfig, Environment, Model
from azureml.core.run import Run

# Set up config of workspace and datastore

aml_workspace = Run.get_context().experiment.workspace
datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')

def generic_01_tpr(sql_account_list: str) :
    query_tpr_aggregated = DataPath(datastore, "SELECT A3 \
    , CASE WHEN employees IS NULL THEN 0 ELSE 1 END AS tpr_match \
    , CASE WHEN scheme_start_year IS NOT NULL THEN scheme_start_year \
    WHEN A1=0 and company_type='P' THEN 2008 \
    WHEN A1=1 and company_type='C' THEN 2004 \
    WHEN A1=1 and company_type='F' THEN 2009 \
    WHEN A1=1 and company_type='E' THEN 2008 \
    WHEN A1=0 and company_type='I' THEN 2012 \
    WHEN A1=0 and company_type='C' THEN 2012 \
    WHEN A1=1 and company_type='P' THEN 1999 \
    WHEN A1=0 and company_type='L' THEN 2007 \
    WHEN A1=1 and company_type='S' THEN 2003 \
    WHEN A1=1 and company_type='I' THEN 2001 \
    WHEN A1=0 and company_type='F' THEN 2008 \
    WHEN A1=1 and company_type='L' THEN 2010 \
    WHEN A1=0 THEN 2011 \
    WHEN A1=1 THEN 2004 \
    ELSE NULL END AS scheme_start_year \
    , CASE WHEN employees IS NOT NULL THEN employees \
    WHEN A1=0 and company_type='C' THEN 16 \
    WHEN A1=0 and company_type='F' THEN 25 \
    WHEN A1=0 and company_type='I' THEN 8 \
    WHEN A1=0 and company_type='L' THEN 34 \
    WHEN A1=0 and company_type='P' THEN 28.5 \
    WHEN A1=0 and company_type='S' THEN 33 \
    WHEN A1=1 and company_type='C' THEN 189 \
    WHEN A1=1 and company_type='E' THEN 259.5 \
    WHEN A1=1 and company_type='F' THEN 156 \
    WHEN A1=1 and company_type='I' THEN 153 \
    WHEN A1=1 and company_type='L' THEN 261 \
    WHEN A1=1 and company_type='P' THEN 683 \
    WHEN A1=1 and company_type='S' THEN 579 \
    WHEN A1=1 THEN 200 \
    WHEN A1=0 THEN 16 \
    ELSE NULL END AS employees \
    , COALESCE(company_type,'X') as company_type \
    , company_status \
    FROM ( \
    SELECT A3 \
    , A1 \
    , MAX(D10) AS employees \
    , MIN(YEAR(D6)) AS scheme_start_year \
    , MAX(SUBSTRING(D12,1,1)) AS company_type \
    , MAX(D8) as company_status \
    FROM  \
    (SELECT A3 \
    , A1 \
    FROM PDS_AI.PT_A \
    WHERE A2<'2022-01-01' AND A3 in ({0}) \
    ) A  \
    LEFT JOIN  \
    (SELECT D15, D10, D6, D12, D8 \
    FROM PDS_AI.PT_D \
    WHERE D15 in ({0}) \
    ) B  \
    ON A.A3=B.D15 \
    GROUP BY A3, A1 \
    ) c".format(sql_account_list))
    tabular_tpr_aggregated = Dataset.Tabular.from_sql_query(query_tpr_aggregated, query_timeout=3600)
    tpr_aggregated = tabular_tpr_aggregated.to_pandas_dataframe()
    
    return tpr_aggregated


def generic_02_sic(sql_account_list: str) :
    query_sic_aggrgated = DataPath(datastore, "SELECT d15 \
    , new_sic_code \
    , sic_division \
    , CASE WHEN 1*sic_division<=3 THEN 'A' \
    WHEN 1*sic_division<=9 THEN 'B' \
    WHEN 1*sic_division<=33 THEN 'C' \
    WHEN 1*sic_division<=35 THEN 'D' \
    WHEN 1*sic_division<=39 THEN 'E' \
    WHEN 1*sic_division<=43 THEN 'F' \
    WHEN 1*sic_division<=47 THEN 'G' \
    WHEN 1*sic_division<=53 THEN 'H' \
    WHEN 1*sic_division<=56 THEN 'I' \
    WHEN 1*sic_division<=63 THEN 'J' \
    WHEN 1*sic_division<=66 THEN 'K' \
    WHEN 1*sic_division<=68 THEN 'L' \
    WHEN 1*sic_division<=75 THEN 'M' \
    WHEN 1*sic_division<=82 THEN 'N' \
    WHEN 1*sic_division<=84 THEN 'O' \
    WHEN 1*sic_division<=85 THEN 'P' \
    WHEN 1*sic_division<=88 THEN 'Q' \
    WHEN 1*sic_division<=93 THEN 'R' \
    WHEN 1*sic_division<=95 THEN 'S' \
    WHEN 1*sic_division<=98 THEN 'T' \
    WHEN 1*sic_division<=99 THEN 'U' \
    ELSE 'Z' END AS sic_section \
    FROM ( \
    SELECT d15 \
    , new_sic_code \
    , SUBSTRING(new_sic_code,1,2) AS sic_division \
    FROM \
    (SELECT d15 \
    , CASE WHEN sic_code='0111' THEN '01110' \
    WHEN sic_code='0112' THEN '01110' \
    WHEN sic_code='0113' THEN '01210' \
    WHEN sic_code='0121' THEN '01410' \
    WHEN sic_code='0122' THEN '01430' \
    WHEN sic_code='0123' THEN '01460' \
    WHEN sic_code='0124' THEN '01470' \
    WHEN sic_code='0125' THEN '01440' \
    WHEN sic_code='0130' THEN '01500' \
    WHEN sic_code='0141' THEN '01610' \
    WHEN sic_code='0142' THEN '01620' \
    WHEN sic_code='0150' THEN '01700' \
    WHEN sic_code='0201' THEN '01290' \
    WHEN sic_code='0202' THEN '02400' \
    WHEN sic_code='0501' THEN '03110' \
    WHEN sic_code='0502' THEN '03210' \
    WHEN sic_code='1010' THEN '05100' \
    WHEN sic_code='1020' THEN '05200' \
    WHEN sic_code='1030' THEN '08920' \
    WHEN sic_code='1110' THEN '06100' \
    WHEN sic_code='1120' THEN '09100' \
    WHEN sic_code='1200' THEN '07210' \
    WHEN sic_code='1310' THEN '07100' \
    WHEN sic_code='1320' THEN '07290' \
    WHEN sic_code='1411' THEN '08110' \
    WHEN sic_code='1412' THEN '08110' \
    WHEN sic_code='1413' THEN '08110' \
    WHEN sic_code='1421' THEN '08120' \
    WHEN sic_code='1422' THEN '08120' \
    WHEN sic_code='1430' THEN '08910' \
    WHEN sic_code='1440' THEN '08930' \
    WHEN sic_code='1450' THEN '08990' \
    WHEN sic_code='1511' THEN '10110' \
    WHEN sic_code='1512' THEN '10110' \
    WHEN sic_code='1513' THEN '10130' \
    WHEN sic_code='1520' THEN '10200' \
    WHEN sic_code='1531' THEN '10310' \
    WHEN sic_code='1532' THEN '10320' \
    WHEN sic_code='1533' THEN '10390' \
    WHEN sic_code='1541' THEN '10410' \
    WHEN sic_code='1542' THEN '10410' \
    WHEN sic_code='1543' THEN '10420' \
    WHEN sic_code='1551' THEN '10510' \
    WHEN sic_code='1552' THEN '10520' \
    WHEN sic_code='1561' THEN '10610' \
    WHEN sic_code='1562' THEN '10620' \
    WHEN sic_code='1571' THEN '10910' \
    WHEN sic_code='1572' THEN '10920' \
    WHEN sic_code='1581' THEN '10710' \
    WHEN sic_code='1582' THEN '10720' \
    WHEN sic_code='1583' THEN '10810' \
    WHEN sic_code='1584' THEN '10820' \
    WHEN sic_code='1585' THEN '10730' \
    WHEN sic_code='1586' THEN '10830' \
    WHEN sic_code='1587' THEN '10840' \
    WHEN sic_code='1588' THEN '10860' \
    WHEN sic_code='1589' THEN '10890' \
    WHEN sic_code='1591' THEN '11010' \
    WHEN sic_code='1592' THEN '11010' \
    WHEN sic_code='1593' THEN '11020' \
    WHEN sic_code='1594' THEN '11030' \
    WHEN sic_code='1595' THEN '11040' \
    WHEN sic_code='1596' THEN '11050' \
    WHEN sic_code='1597' THEN '11060' \
    WHEN sic_code='1598' THEN '11070' \
    WHEN sic_code='1600' THEN '12000' \
    WHEN sic_code='1711' THEN '13100' \
    WHEN sic_code='1712' THEN '13100' \
    WHEN sic_code='1713' THEN '13100' \
    WHEN sic_code='1714' THEN '13100' \
    WHEN sic_code='1715' THEN '13100' \
    WHEN sic_code='1716' THEN '13100' \
    WHEN sic_code='1717' THEN '13100' \
    WHEN sic_code='1721' THEN '13200' \
    WHEN sic_code='1722' THEN '13200' \
    WHEN sic_code='1723' THEN '13200' \
    WHEN sic_code='1724' THEN '13200' \
    WHEN sic_code='1725' THEN '13200' \
    WHEN sic_code='1730' THEN '13300' \
    WHEN sic_code='1740' THEN '13920' \
    WHEN sic_code='1751' THEN '13930' \
    WHEN sic_code='1752' THEN '13940' \
    WHEN sic_code='1753' THEN '13950' \
    WHEN sic_code='1754' THEN '13960' \
    WHEN sic_code='1760' THEN '13910' \
    WHEN sic_code='1771' THEN '14190' \
    WHEN sic_code='1772' THEN '14390' \
    WHEN sic_code='1810' THEN '14110' \
    WHEN sic_code='1821' THEN '14120' \
    WHEN sic_code='1822' THEN '14130' \
    WHEN sic_code='1823' THEN '14140' \
    WHEN sic_code='1824' THEN '14190' \
    WHEN sic_code='1830' THEN '14200' \
    WHEN sic_code='1910' THEN '15110' \
    WHEN sic_code='1920' THEN '15120' \
    WHEN sic_code='1930' THEN '15200' \
    WHEN sic_code='2010' THEN '16100' \
    WHEN sic_code='2020' THEN '16210' \
    WHEN sic_code='2030' THEN '16220' \
    WHEN sic_code='2040' THEN '16240' \
    WHEN sic_code='2051' THEN '16290' \
    WHEN sic_code='2052' THEN '16290' \
    WHEN sic_code='2111' THEN '17110' \
    WHEN sic_code='2112' THEN '17120' \
    WHEN sic_code='2121' THEN '17210' \
    WHEN sic_code='2122' THEN '17220' \
    WHEN sic_code='2123' THEN '17230' \
    WHEN sic_code='2124' THEN '17240' \
    WHEN sic_code='2125' THEN '17290' \
    WHEN sic_code='2211' THEN '32990' \
    WHEN sic_code='2212' THEN '58130' \
    WHEN sic_code='2213' THEN '58140' \
    WHEN sic_code='2214' THEN '59200' \
    WHEN sic_code='2215' THEN '58190' \
    WHEN sic_code='2221' THEN '18110' \
    WHEN sic_code='2222' THEN '17230' \
    WHEN sic_code='2223' THEN '18140' \
    WHEN sic_code='2224' THEN '18130' \
    WHEN sic_code='2225' THEN '18130' \
    WHEN sic_code='2231' THEN '18200' \
    WHEN sic_code='2232' THEN '18200' \
    WHEN sic_code='2233' THEN '18200' \
    WHEN sic_code='2310' THEN '19100' \
    WHEN sic_code='2320' THEN '19200' \
    WHEN sic_code='2330' THEN '20130' \
    WHEN sic_code='2411' THEN '20110' \
    WHEN sic_code='2412' THEN '20120' \
    WHEN sic_code='2413' THEN '20130' \
    WHEN sic_code='2414' THEN '19100' \
    WHEN sic_code='2415' THEN '20150' \
    WHEN sic_code='2416' THEN '20160' \
    WHEN sic_code='2417' THEN '20170' \
    WHEN sic_code='2420' THEN '20200' \
    WHEN sic_code='2430' THEN '20300' \
    WHEN sic_code='2441' THEN '21100' \
    WHEN sic_code='2442' THEN '21200' \
    WHEN sic_code='2451' THEN '20410' \
    WHEN sic_code='2452' THEN '20420' \
    WHEN sic_code='2461' THEN '20510' \
    WHEN sic_code='2462' THEN '20520' \
    WHEN sic_code='2463' THEN '20530' \
    WHEN sic_code='2464' THEN '20590' \
    WHEN sic_code='2465' THEN '26800' \
    WHEN sic_code='2466' THEN '20590' \
    WHEN sic_code='2470' THEN '20600' \
    WHEN sic_code='2511' THEN '22110' \
    WHEN sic_code='2512' THEN '22110' \
    WHEN sic_code='2513' THEN '22190' \
    WHEN sic_code='2521' THEN '22210' \
    WHEN sic_code='2522' THEN '22220' \
    WHEN sic_code='2523' THEN '22230' \
    WHEN sic_code='2524' THEN '22290' \
    WHEN sic_code='2611' THEN '23110' \
    WHEN sic_code='2612' THEN '23120' \
    WHEN sic_code='2613' THEN '23130' \
    WHEN sic_code='2614' THEN '23140' \
    WHEN sic_code='2615' THEN '23190' \
    WHEN sic_code='2621' THEN '23410' \
    WHEN sic_code='2622' THEN '23420' \
    WHEN sic_code='2623' THEN '23430' \
    WHEN sic_code='2624' THEN '23440' \
    WHEN sic_code='2625' THEN '23490' \
    WHEN sic_code='2626' THEN '23200' \
    WHEN sic_code='2630' THEN '23310' \
    WHEN sic_code='2640' THEN '23320' \
    WHEN sic_code='2651' THEN '23510' \
    WHEN sic_code='2652' THEN '23520' \
    WHEN sic_code='2653' THEN '23520' \
    WHEN sic_code='2661' THEN '23610' \
    WHEN sic_code='2662' THEN '23620' \
    WHEN sic_code='2663' THEN '23630' \
    WHEN sic_code='2664' THEN '23640' \
    WHEN sic_code='2665' THEN '23650' \
    WHEN sic_code='2666' THEN '23690' \
    WHEN sic_code='2670' THEN '23700' \
    WHEN sic_code='2681' THEN '23910' \
    WHEN sic_code='2682' THEN '23990' \
    WHEN sic_code='2710' THEN '24100' \
    WHEN sic_code='2721' THEN '24510' \
    WHEN sic_code='2722' THEN '24200' \
    WHEN sic_code='2731' THEN '24310' \
    WHEN sic_code='2732' THEN '24320' \
    WHEN sic_code='2733' THEN '24330' \
    WHEN sic_code='2734' THEN '24340' \
    WHEN sic_code='2741' THEN '24410' \
    WHEN sic_code='2742' THEN '24420' \
    WHEN sic_code='2743' THEN '24430' \
    WHEN sic_code='2744' THEN '24440' \
    WHEN sic_code='2745' THEN '24450' \
    WHEN sic_code='2751' THEN '24510' \
    WHEN sic_code='2752' THEN '24520' \
    WHEN sic_code='2753' THEN '24530' \
    WHEN sic_code='2754' THEN '24540' \
    WHEN sic_code='2811' THEN '24330' \
    WHEN sic_code='2812' THEN '25120' \
    WHEN sic_code='2821' THEN '25290' \
    WHEN sic_code='2822' THEN '25210' \
    WHEN sic_code='2830' THEN '25300' \
    WHEN sic_code='2840' THEN '25500' \
    WHEN sic_code='2851' THEN '25610' \
    WHEN sic_code='2852' THEN '25620' \
    WHEN sic_code='2861' THEN '25710' \
    WHEN sic_code='2862' THEN '25730' \
    WHEN sic_code='2863' THEN '25720' \
    WHEN sic_code='2871' THEN '25910' \
    WHEN sic_code='2872' THEN '25920' \
    WHEN sic_code='2873' THEN '25930' \
    WHEN sic_code='2874' THEN '25930' \
    WHEN sic_code='2875' THEN '25710' \
    WHEN sic_code='2911' THEN '28110' \
    WHEN sic_code='2912' THEN '28110' \
    WHEN sic_code='2913' THEN '28120' \
    WHEN sic_code='2914' THEN '28120' \
    WHEN sic_code='2921' THEN '28210' \
    WHEN sic_code='2922' THEN '28220' \
    WHEN sic_code='2923' THEN '28250' \
    WHEN sic_code='2924' THEN '28290' \
    WHEN sic_code='2931' THEN '28300' \
    WHEN sic_code='2932' THEN '28300' \
    WHEN sic_code='2941' THEN '28240' \
    WHEN sic_code='2942' THEN '28410' \
    WHEN sic_code='2943' THEN '27900' \
    WHEN sic_code='2951' THEN '28910' \
    WHEN sic_code='2952' THEN '28920' \
    WHEN sic_code='2953' THEN '28300' \
    WHEN sic_code='2954' THEN '28940' \
    WHEN sic_code='2955' THEN '28950' \
    WHEN sic_code='2956' THEN '25730' \
    WHEN sic_code='2960' THEN '25400' \
    WHEN sic_code='2971' THEN '27510' \
    WHEN sic_code='2972' THEN '27520' \
    WHEN sic_code='3001' THEN '28230' \
    WHEN sic_code='3002' THEN '26200' \
    WHEN sic_code='3110' THEN '26110' \
    WHEN sic_code='3120' THEN '26110' \
    WHEN sic_code='3130' THEN '26110' \
    WHEN sic_code='3140' THEN '27200' \
    WHEN sic_code='3150' THEN '27400' \
    WHEN sic_code='3161' THEN '27400' \
    WHEN sic_code='3162' THEN '23440' \
    WHEN sic_code='3210' THEN '26110' \
    WHEN sic_code='3220' THEN '26300' \
    WHEN sic_code='3230' THEN '26110' \
    WHEN sic_code='3310' THEN '26600' \
    WHEN sic_code='3320' THEN '26510' \
    WHEN sic_code='3330' THEN '26510' \
    WHEN sic_code='3340' THEN '26700' \
    WHEN sic_code='3350' THEN '26520' \
    WHEN sic_code='3410' THEN '28920' \
    WHEN sic_code='3420' THEN '29200' \
    WHEN sic_code='3430' THEN '28110' \
    WHEN sic_code='3511' THEN '30110' \
    WHEN sic_code='3512' THEN '30120' \
    WHEN sic_code='3520' THEN '30200' \
    WHEN sic_code='3530' THEN '28990' \
    WHEN sic_code='3541' THEN '30910' \
    WHEN sic_code='3542' THEN '30920' \
    WHEN sic_code='3543' THEN '30920' \
    WHEN sic_code='3550' THEN '28220' \
    WHEN sic_code='3611' THEN '29320' \
    WHEN sic_code='3612' THEN '28230' \
    WHEN sic_code='3613' THEN '31020' \
    WHEN sic_code='3614' THEN '31090' \
    WHEN sic_code='3615' THEN '31030' \
    WHEN sic_code='3621' THEN '32110' \
    WHEN sic_code='3622' THEN '32120' \
    WHEN sic_code='3630' THEN '32200' \
    WHEN sic_code='3640' THEN '32300' \
    WHEN sic_code='3650' THEN '26400' \
    WHEN sic_code='3661' THEN '32130' \
    WHEN sic_code='3662' THEN '22190' \
    WHEN sic_code='3663' THEN '13990' \
    WHEN sic_code='3710' THEN '38310' \
    WHEN sic_code='3720' THEN '38320' \
    WHEN sic_code='4011' THEN '35110' \
    WHEN sic_code='4012' THEN '35120' \
    WHEN sic_code='4013' THEN '35130' \
    WHEN sic_code='4021' THEN '35210' \
    WHEN sic_code='4022' THEN '35220' \
    WHEN sic_code='4030' THEN '35300' \
    WHEN sic_code='4100' THEN '36000' \
    WHEN sic_code='4511' THEN '43110' \
    WHEN sic_code='4512' THEN '43130' \
    WHEN sic_code='4521' THEN '41200' \
    WHEN sic_code='4522' THEN '43910' \
    WHEN sic_code='4523' THEN '41200' \
    WHEN sic_code='4524' THEN '42210' \
    WHEN sic_code='4525' THEN '42210' \
    WHEN sic_code='4531' THEN '43210' \
    WHEN sic_code='4532' THEN '43290' \
    WHEN sic_code='4533' THEN '43220' \
    WHEN sic_code='4534' THEN '43210' \
    WHEN sic_code='4541' THEN '43310' \
    WHEN sic_code='4542' THEN '43320' \
    WHEN sic_code='4543' THEN '43330' \
    WHEN sic_code='4544' THEN '43340' \
    WHEN sic_code='4545' THEN '43390' \
    WHEN sic_code='4550' THEN '43990' \
    WHEN sic_code='5010' THEN '45110' \
    WHEN sic_code='5020' THEN '45200' \
    WHEN sic_code='5030' THEN '45310' \
    WHEN sic_code='5040' THEN '45400' \
    WHEN sic_code='5050' THEN '47300' \
    WHEN sic_code='5111' THEN '46110' \
    WHEN sic_code='5112' THEN '46120' \
    WHEN sic_code='5113' THEN '46130' \
    WHEN sic_code='5114' THEN '46140' \
    WHEN sic_code='5115' THEN '46150' \
    WHEN sic_code='5116' THEN '46160' \
    WHEN sic_code='5117' THEN '46170' \
    WHEN sic_code='5118' THEN '46180' \
    WHEN sic_code='5119' THEN '46190' \
    WHEN sic_code='5121' THEN '46210' \
    WHEN sic_code='5122' THEN '46220' \
    WHEN sic_code='5123' THEN '46230' \
    WHEN sic_code='5124' THEN '46240' \
    WHEN sic_code='5125' THEN '46210' \
    WHEN sic_code='5131' THEN '10390' \
    WHEN sic_code='5132' THEN '46320' \
    WHEN sic_code='5133' THEN '46330' \
    WHEN sic_code='5134' THEN '11010' \
    WHEN sic_code='5135' THEN '46350' \
    WHEN sic_code='5136' THEN '46360' \
    WHEN sic_code='5137' THEN '46370' \
    WHEN sic_code='5138' THEN '46310' \
    WHEN sic_code='5139' THEN '46390' \
    WHEN sic_code='5141' THEN '46410' \
    WHEN sic_code='5142' THEN '46420' \
    WHEN sic_code='5143' THEN '46430' \
    WHEN sic_code='5144' THEN '46440' \
    WHEN sic_code='5145' THEN '46450' \
    WHEN sic_code='5146' THEN '46460' \
    WHEN sic_code='5147' THEN '46430' \
    WHEN sic_code='5151' THEN '46710' \
    WHEN sic_code='5152' THEN '46720' \
    WHEN sic_code='5153' THEN '46730' \
    WHEN sic_code='5154' THEN '46740' \
    WHEN sic_code='5155' THEN '46750' \
    WHEN sic_code='5156' THEN '46760' \
    WHEN sic_code='5157' THEN '46770' \
    WHEN sic_code='5181' THEN '46620' \
    WHEN sic_code='5182' THEN '46630' \
    WHEN sic_code='5183' THEN '46640' \
    WHEN sic_code='5184' THEN '46510' \
    WHEN sic_code='5185' THEN '46650' \
    WHEN sic_code='5186' THEN '46520' \
    WHEN sic_code='5187' THEN '46690' \
    WHEN sic_code='5188' THEN '46610' \
    WHEN sic_code='5190' THEN '46900' \
    WHEN sic_code='5211' THEN '47110' \
    WHEN sic_code='5212' THEN '47190' \
    WHEN sic_code='5221' THEN '47210' \
    WHEN sic_code='5222' THEN '47220' \
    WHEN sic_code='5223' THEN '47230' \
    WHEN sic_code='5224' THEN '47240' \
    WHEN sic_code='5225' THEN '47250' \
    WHEN sic_code='5226' THEN '47260' \
    WHEN sic_code='5227' THEN '47210' \
    WHEN sic_code='5231' THEN '47730' \
    WHEN sic_code='5232' THEN '47740' \
    WHEN sic_code='5233' THEN '47750' \
    WHEN sic_code='5241' THEN '47510' \
    WHEN sic_code='5242' THEN '47710' \
    WHEN sic_code='5243' THEN '47720' \
    WHEN sic_code='5244' THEN '47530' \
    WHEN sic_code='5245' THEN '47430' \
    WHEN sic_code='5246' THEN '47520' \
    WHEN sic_code='5247' THEN '47610' \
    WHEN sic_code='5248' THEN '47410' \
    WHEN sic_code='5250' THEN '47790' \
    WHEN sic_code='5261' THEN '47910' \
    WHEN sic_code='5262' THEN '47810' \
    WHEN sic_code='5263' THEN '47790' \
    WHEN sic_code='5271' THEN '95230' \
    WHEN sic_code='5272' THEN '95210' \
    WHEN sic_code='5273' THEN '95250' \
    WHEN sic_code='5274' THEN '13300' \
    WHEN sic_code='5510' THEN '55100' \
    WHEN sic_code='5521' THEN '55200' \
    WHEN sic_code='5522' THEN '55300' \
    WHEN sic_code='5523' THEN '55200' \
    WHEN sic_code='5530' THEN '56100' \
    WHEN sic_code='5540' THEN '56300' \
    WHEN sic_code='5551' THEN '56290' \
    WHEN sic_code='5552' THEN '56210' \
    WHEN sic_code='6010' THEN '49100' \
    WHEN sic_code='6021' THEN '49310' \
    WHEN sic_code='6022' THEN '49320' \
    WHEN sic_code='6023' THEN '49390' \
    WHEN sic_code='6024' THEN '49410' \
    WHEN sic_code='6030' THEN '49500' \
    WHEN sic_code='6110' THEN '50100' \
    WHEN sic_code='6120' THEN '50300' \
    WHEN sic_code='6210' THEN '51100' \
    WHEN sic_code='6220' THEN '51100' \
    WHEN sic_code='6230' THEN '51220' \
    WHEN sic_code='6311' THEN '52240' \
    WHEN sic_code='6312' THEN '52100' \
    WHEN sic_code='6321' THEN '52210' \
    WHEN sic_code='6322' THEN '52220' \
    WHEN sic_code='6323' THEN '52230' \
    WHEN sic_code='6330' THEN '79110' \
    WHEN sic_code='6340' THEN '52290' \
    WHEN sic_code='6411' THEN '53100' \
    WHEN sic_code='6412' THEN '53200' \
    WHEN sic_code='6420' THEN '60100' \
    WHEN sic_code='6511' THEN '64110' \
    WHEN sic_code='6512' THEN '64190' \
    WHEN sic_code='6521' THEN '64910' \
    WHEN sic_code='6522' THEN '64920' \
    WHEN sic_code='6523' THEN '64200' \
    WHEN sic_code='6601' THEN '65110' \
    WHEN sic_code='6602' THEN '65200' \
    WHEN sic_code='6603' THEN '65120' \
    WHEN sic_code='6711' THEN '66110' \
    WHEN sic_code='6712' THEN '66120' \
    WHEN sic_code='6713' THEN '66120' \
    WHEN sic_code='6720' THEN '66210' \
    WHEN sic_code='7011' THEN '41100' \
    WHEN sic_code='7012' THEN '68100' \
    WHEN sic_code='7020' THEN '68200' \
    WHEN sic_code='7031' THEN '68310' \
    WHEN sic_code='7032' THEN '68320' \
    WHEN sic_code='7110' THEN '77110' \
    WHEN sic_code='7121' THEN '77120' \
    WHEN sic_code='7122' THEN '77340' \
    WHEN sic_code='7123' THEN '77350' \
    WHEN sic_code='7131' THEN '77310' \
    WHEN sic_code='7132' THEN '77320' \
    WHEN sic_code='7133' THEN '77330' \
    WHEN sic_code='7134' THEN '77390' \
    WHEN sic_code='7140' THEN '77210' \
    WHEN sic_code='7210' THEN '62020' \
    WHEN sic_code='7221' THEN '58210' \
    WHEN sic_code='7222' THEN '62010' \
    WHEN sic_code='7230' THEN '62030' \
    WHEN sic_code='7240' THEN '58110' \
    WHEN sic_code='7250' THEN '33120' \
    WHEN sic_code='7260' THEN '62090' \
    WHEN sic_code='7310' THEN '72110' \
    WHEN sic_code='7320' THEN '72200' \
    WHEN sic_code='7411' THEN '69100' \
    WHEN sic_code='7412' THEN '69200' \
    WHEN sic_code='7413' THEN '73200' \
    WHEN sic_code='7414' THEN '02400' \
    WHEN sic_code='7415' THEN '64200' \
    WHEN sic_code='7420' THEN '71110' \
    WHEN sic_code='7430' THEN '71200' \
    WHEN sic_code='7440' THEN '73110' \
    WHEN sic_code='7450' THEN '78100' \
    WHEN sic_code='7460' THEN '74900' \
    WHEN sic_code='7470' THEN '81210' \
    WHEN sic_code='7481' THEN '74200' \
    WHEN sic_code='7482' THEN '82920' \
    WHEN sic_code='7485' THEN '74300' \
    WHEN sic_code='7486' THEN '82200' \
    WHEN sic_code='7487' THEN '59200' \
    WHEN sic_code='7511' THEN '84110' \
    WHEN sic_code='7512' THEN '84120' \
    WHEN sic_code='7513' THEN '82990' \
    WHEN sic_code='7514' THEN '81100' \
    WHEN sic_code='7521' THEN '84210' \
    WHEN sic_code='7522' THEN '84220' \
    WHEN sic_code='7523' THEN '84230' \
    WHEN sic_code='7524' THEN '84240' \
    WHEN sic_code='7525' THEN '84250' \
    WHEN sic_code='7530' THEN '84300' \
    WHEN sic_code='8010' THEN '85100' \
    WHEN sic_code='8021' THEN '85310' \
    WHEN sic_code='8022' THEN '85320' \
    WHEN sic_code='8030' THEN '85410' \
    WHEN sic_code='8041' THEN '85530' \
    WHEN sic_code='8042' THEN '85320' \
    WHEN sic_code='8511' THEN '86100' \
    WHEN sic_code='8512' THEN '86210' \
    WHEN sic_code='8513' THEN '86230' \
    WHEN sic_code='8514' THEN '86900' \
    WHEN sic_code='8520' THEN '75000' \
    WHEN sic_code='8531' THEN '87200' \
    WHEN sic_code='8532' THEN '88100' \
    WHEN sic_code='9001' THEN '37000' \
    WHEN sic_code='9002' THEN '38110' \
    WHEN sic_code='9003' THEN '38110' \
    WHEN sic_code='9111' THEN '94110' \
    WHEN sic_code='9112' THEN '94120' \
    WHEN sic_code='9120' THEN '94200' \
    WHEN sic_code='9131' THEN '94910' \
    WHEN sic_code='9132' THEN '94920' \
    WHEN sic_code='9133' THEN '94990' \
    WHEN sic_code='9211' THEN '59110' \
    WHEN sic_code='9212' THEN '59130' \
    WHEN sic_code='9213' THEN '59140' \
    WHEN sic_code='9220' THEN '59110' \
    WHEN sic_code='9231' THEN '90010' \
    WHEN sic_code='9232' THEN '79900' \
    WHEN sic_code='9233' THEN '93210' \
    WHEN sic_code='9234' THEN '79900' \
    WHEN sic_code='9240' THEN '63910' \
    WHEN sic_code='9251' THEN '91010' \
    WHEN sic_code='9252' THEN '91020' \
    WHEN sic_code='9253' THEN '91040' \
    WHEN sic_code='9261' THEN '93110' \
    WHEN sic_code='9262' THEN '79900' \
    WHEN sic_code='9271' THEN '92000' \
    WHEN sic_code='9272' THEN '01620' \
    WHEN sic_code='9301' THEN '96010' \
    WHEN sic_code='9302' THEN '96020' \
    WHEN sic_code='9303' THEN '96030' \
    WHEN sic_code='9304' THEN '93130' \
    WHEN sic_code='9305' THEN '85510' \
    WHEN sic_code='9500' THEN '97000' \
    WHEN sic_code='9600' THEN '98100' \
    WHEN sic_code='9700' THEN '98200' \
    WHEN sic_code='9900' THEN '99000' \
    ELSE sic_code END AS new_sic_code \
    FROM ( \
    SELECT d15, MAX(SUBSTRING(I4,1,CHARINDEX(' ',i4)-1)) AS sic_code \
    FROM \
    (SELECT MAX(d15) AS d15, d3 \
    FROM PDS_AI.PT_D \
    WHERE d15 is not null and d15>-1 AND D15 in ({0}) \
    GROUP BY d3) a \
    INNER JOIN (SELECT i1 ,i4 FROM PDS_AI.PT_I) b \
    ON a.d3=b.i1 \
    GROUP BY d15 \
    ) c) d) e".format(sql_account_list))
    tabular_sic_aggregated = Dataset.Tabular.from_sql_query(query_sic_aggregated, query_timeout=3600)
    sic_aggregated = tabular_sic_aggregated.to_pandas_dataframe()
    
    return sic_aggregated
