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
    ON c.vanid = e.vanid
    WHERE e.rn = 1
), ea_addrs AS (
    SELECT
        c.vanid,
        a.state
    FROM
        indivisible_ea.contacts_iv c
    JOIN (
        SELECT
            vanid,
            a.state,
            ROW_NUMBER() OVER(PARTITION BY vanid ORDER BY datemodified DESC) as rn
        FROM
            indivisible_ea.contactsaddresses_iv a
    ) a
        ON a.vanid = c.vanid
        WHERE a.rn = 1
), ea_phones AS (
    SELECT vanid, phone
    FROM (
        SELECT
            vanid,
            phone,
            ROW_NUMBER() OVER(PARTITION BY vanid ORDER BY datemodified DESC) as rn
        FROM indivisible_ea.contactsphones_iv
        WHERE datesuppressed IS NULL
    )
    WHERE rn = 1
), ea_xwalks AS (
    SELECT
        voterbase_id as xwalk_id,
        LOWER(COALESCE(ea_c.firstname, '')) AS first_name_field,
        LOWER(COALESCE(ea_c.lastname, '')) AS last_name_field,
        LOWER(COALESCE(ea_emails.email, '')) AS email_field,
        LOWER(COALESCE(a.state, '')) as state_field,
        LOWER(COALESCE(p.phone, '')) as raw_phone_field
    FROM
        indivisible_ea.contacts_iv ea_c
    LEFT JOIN ea_emails
        ON ea_emails.vanid = ea_c.vanid
    LEFT JOIN ea_addrs a
        ON a.vanid = ea_c.vanid
    LEFT JOIN ea_phones p
        ON p.vanid = ea_c.vanid
    JOIN indivisible_infrastructure.xwalk x
        ON ea_c.vanid = x.everyaction_id
), ak_phones AS (
    SELECT user_id, phone
    FROM (
        SELECT
            user_id,
            phone,
            ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY updated_at DESC) as rn
        FROM indivisible_ak.core_phone
    )
    WHERE rn = 1
    
), ak_xwalks AS (
    SELECT
        voterbase_id as xwalk_id,
        LOWER(COALESCE(ak.first_name, '')) AS first_name_field,
        LOWER(COALESCE(ak.last_name, '')) AS last_name_field,
        LOWER(COALESCE(ak.email, '')) AS email_field,
        LOWER(COALESCE(ak.state, '')) as state_field,
        LOWER(COALESCE(ak_phones.phone, '')) as raw_phone_field
    FROM
        indivisible_ak.core_user ak
    JOIN indivisible_infrastructure.xwalk x
        ON ak.id = x.actionkit_id
    LEFT JOIN ak_phones
        on ak_phones.user_id = ak.id
), mc_xwalks AS (
    SELECT
        voterbase_id as xwalk_id,
        LOWER(COALESCE(mc.first_name, '')) AS first_name_field,
        LOWER(COALESCE(mc.last_name, '')) AS last_name_field,
        LOWER(COALESCE(mc.email, '')) AS email_field,
        LOWER(COALESCE(mc.address_state, '')) as state_field,
        LOWER(COALESCE(mc.phone_number, '')) as raw_phone_field
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
        LOWER(COALESCE(ab.state, '')) as state_field,
        LOWER(COALESCE(ab.phone, '')) as raw_phone_field
    FROM
        tmc_ab.in_donations ab
    JOIN indivisible_infrastructure.xwalk x
        ON ab.ordernumber = x.actblue_id
), combined_xwalks AS (
    SELECT 
        xwalk_id,
        first_name_field,
        last_name_field,
        email_field,
        state_field,
        RIGHT(REGEXP_REPLACE(raw_phone_field, '[^0-9]', ''), 10) AS phone_field,
        first_name_field || '|' || last_name_field || '|' || email_field || '|' || phone_field || '|' || state_field as comp_string
    FROM (
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
)
SELECT distinct
    c1.first_name_field AS first_name1, 
    c1.last_name_field AS last_name1,
    c1.email_field AS email1,
    c1.phone_field AS phone1,
    c1.state_field AS state1,
    c2.first_name_field AS first_name2,
    c2.last_name_field AS last_name2,
    c2.email_field AS email2,
    c2.phone_field AS phone2,
    c2.state_field AS state2,
    1 as label
FROM 
    combined_xwalks c1
JOIN 
    combined_xwalks c2
ON 
    c1.xwalk_id = c2.xwalk_id
WHERE 
    c1.comp_string <> c2.comp_string AND
    -- Prevent duplicate pairs (1,2) then (2,1) later from showing up
    c1.comp_string < c2.comp_string;