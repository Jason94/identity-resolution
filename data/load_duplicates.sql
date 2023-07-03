WITH ea_emails AS (
    SELECT 
        c.vanid,
        e.email
    FROM 
        indivisible_ea.contacts_iv c
    JOIN 
    (
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
), results AS (
    SELECT 
        LOWER(ea_c.firstname) AS ea_first_name,
        LOWER(ea_c.lastname) AS ea_last_name,
        LOWER(ea_emails.email) AS ea_email,
        LOWER(ak_u.first_name) AS ak_first_name,
        LOWER(ak_u.last_name) AS ak_last_name,
        LOWER(ak_u.email) AS ak_email,
        LOWER(mc.first_name) AS mc_first_name,
        LOWER(mc.last_name) AS mc_last_name,
        LOWER(mc.email) AS mc_email
    FROM indivisible_infrastructure.xwalk x
    LEFT JOIN indivisible_ea.contacts_iv ea_c
        ON ea_c.vanid = x.everyaction_id
    LEFT JOIN ea_emails
        ON ea_emails.vanid = ea_c.vanid
    LEFT JOIN indivisible_ak.core_user ak_u
        ON ak_u.id = x.actionkit_id
    LEFT JOIN in_mobilecommons.profiles mc
        ON mc.id = x.mobilecommons_id
    WHERE 
        LOWER(ea_c.firstname) != LOWER(ak_u.first_name) 
        OR LOWER(ea_c.lastname) != LOWER(ak_u.last_name) 
        OR LOWER(mc.first_name) != LOWER(ea_c.firstname) 
        OR LOWER(mc.last_name) != LOWER(ea_c.lastname) 
        OR LOWER(ak_u.first_name) != LOWER(mc.first_name) 
        OR LOWER(ak_u.last_name) != LOWER(mc.last_name)
), combined AS (
    SELECT ea_first_name AS first_name1, ea_last_name AS last_name1, ea_email AS email1, ak_first_name AS first_name2, ak_last_name AS last_name2, ak_email AS email2
    FROM results
    UNION ALL
    SELECT ea_first_name, ea_last_name, ea_email, mc_first_name, mc_last_name, mc_email
    FROM results
    UNION ALL
    SELECT ak_first_name, ak_last_name, ak_email, mc_first_name, mc_last_name, mc_email
    FROM results
), duplicates_final AS (
    SELECT *, 1 AS label
    FROM combined
    WHERE (first_name1 IS NOT NULL AND first_name2 IS NOT NULL) 
        AND (first_name1 != first_name2 OR last_name1 != last_name2)
)
SELECT *
FROM duplicates_final;