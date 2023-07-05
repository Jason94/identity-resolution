WITH ea_emails AS (
    SELECT 
        c.vanid,
        e.email
    FROM 
        indivisible_ea.contacts_iv c
    JOIN (
        SELECT 
            ce.email, 
            ce.vanid, 
            ce.datecreated, 
            ROW_NUMBER() OVER(PARTITION BY ce.vanid ORDER BY ce.datecreated DESC) AS rn
        FROM
            indivisible_ea.contactsemails_iv ce
        LEFT JOIN 
            indivisible_ea.emailsubscriptions_iv es
        ON 
            LOWER(ce.email) = LOWER(es.email)
        WHERE 
            es.dateunsubscribed IS NULL
    ) e
    ON 
        c.vanid = e.vanid
    WHERE 
        e.rn = 1
), ea_xwalks AS (
    SELECT
        voterbase_id as xwalk_id,
        LOWER(COALESCE(ea_c.firstname, '')) AS first_name_field,
        LOWER(COALESCE(ea_c.lastname, '')) AS last_name_field,
        LOWER(COALESCE(ea_emails.email, '')) AS email_field,
        first_name_field || '|' || last_name_field || '|' || email_field as comp_string
    FROM
        indivisible_ea.contacts_iv ea_c
    LEFT JOIN ea_emails
        ON ea_emails.vanid = ea_c.vanid
    JOIN indivisible_infrastructure.xwalk x
        ON ea_c.vanid = x.everyaction_id
), ak_xwalks AS (
    SELECT
        voterbase_id as xwalk_id,
        LOWER(COALESCE(ak.first_name, '')) AS first_name_field,
        LOWER(COALESCE(ak.last_name, '')) AS last_name_field,
        LOWER(COALESCE(ak.email, '')) AS email_field,
        first_name_field || '|' || last_name_field || '|' || email_field as comp_string
    FROM
        indivisible_ak.core_user ak
    JOIN indivisible_infrastructure.xwalk x
        ON ak.id = x.actionkit_id
), mc_xwalks AS (
    SELECT
        voterbase_id as xwalk_id,
        LOWER(COALESCE(mc.first_name, '')) AS first_name_field,
        LOWER(COALESCE(mc.last_name, '')) AS last_name_field,
        LOWER(COALESCE(mc.email, '')) AS email_field,
        first_name_field || '|' || last_name_field || '|' || email_field as comp_string
    FROM
        in_mobilecommons.profiles mc
    JOIN indivisible_infrastructure.xwalk x
        ON mc.id = x.mobilecommons_id
), ab_xwalks AS (
    SELECT
        voterbase_id as xwalk_id,
        LOWER(COALESCE(ab.firstname, '')) AS first_name_field,
        LOWER(COALESCE(ab.lastname, '')) AS last_name_field,
        LOWER(COALESCE(ab.email, '')) AS email_field,
        first_name_field || '|' || last_name_field || '|' || email_field as comp_string
    FROM
        tmc_ab.in_donations ab
    JOIN indivisible_infrastructure.xwalk x
        ON ab.ordernumber = x.actblue_id
), combined_xwalks AS (
        SELECT *
        FROM ea_xwalks
    UNION ALL
        SELECT *
        FROM ak_xwalks
    UNION ALL
        SELECT *
        FROM mc_xwalks
    UNION ALL
        SELECT *
        FROM ab_xwalks
)
SELECT distinct
    c1.first_name_field AS first_name1, 
    c1.last_name_field AS last_name1,
    c1.email_field AS email1,
    c2.first_name_field AS first_name2,
    c2.last_name_field AS last_name2,
    c2.email_field AS email2,
    1 as label
FROM 
    combined_xwalks c1
JOIN 
    combined_xwalks c2
ON 
    c1.xwalk_id = c2.xwalk_id
WHERE 
    c1.comp_string <> c2.comp_string;