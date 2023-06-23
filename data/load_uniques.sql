WITH ea_c AS (
    SELECT 
        x.tmc_person_id,
        LOWER(e.firstname) AS ea_first_name, 
        LOWER(e.lastname) AS ea_last_name,
        ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS rownum
    FROM indivisible_infrastructure.xwalk x
    INNER JOIN indivisible_ea.contacts_iv e
        ON e.vanid = x.everyaction_id
),
ak_u AS (
    SELECT 
        x.tmc_person_id,
        LOWER(a.first_name) AS ak_first_name, 
        LOWER(a.last_name) AS ak_last_name,
        ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS rownum
    FROM indivisible_infrastructure.xwalk x
    INNER JOIN indivisible_ak.core_user a
        ON a.id = x.actionkit_id
)
SELECT 
    ea_c.ea_first_name, 
    ea_c.ea_last_name, 
    ak_u.ak_first_name, 
    ak_u.ak_last_name
FROM ea_c
INNER JOIN ak_u
    ON ea_c.rownum = ak_u.rownum
WHERE 
    ea_c.tmc_person_id != ak_u.tmc_person_id
LIMIT 7989;