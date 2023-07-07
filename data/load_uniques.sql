with ea_emails AS (
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
), distincts as (
    WITH ea_c AS (
        SELECT 
            x.tmc_person_id,
            LOWER(e.firstname) AS ea_first_name, 
            LOWER(e.lastname) AS ea_last_name,
            LOWER(COALESCE(ea_emails.email, '')) AS ea_email,
            LOWER(COALESCE(ea_addrs.state, '')) as ea_state,
            LOWER(COALESCE(ea_phones.phone, '')) as ea_phone,
            ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS rownum
        FROM indivisible_infrastructure.xwalk x
        INNER JOIN indivisible_ea.contacts_iv e
            ON e.vanid = x.everyaction_id
        LEFT JOIN ea_emails
            ON ea_emails.vanid = e.vanid
        LEFT JOIN ea_addrs
            ON ea_addrs.vanid = e.vanid
        LEFT JOIN ea_phones
            ON ea_phones.vanid = e.vanid
    ),
    ak_phones AS (
        SELECT user_id, phone
        FROM (
            SELECT
                user_id,
                phone,
                ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY updated_at DESC) as rn
            FROM indivisible_ak.core_phone
        )
        WHERE rn = 1
        
    ), ak_u AS (
        SELECT 
            x.tmc_person_id,
            LOWER(a.first_name) AS ak_first_name, 
            LOWER(a.last_name) AS ak_last_name,
            LOWER(a.email) AS ak_email,
            LOWER(COALESCE(a.state, '')) as ak_state,
            LOWER(COALESCE(p.phone, '')) as ak_phone,
            ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS rownum
        FROM indivisible_infrastructure.xwalk x
        INNER JOIN indivisible_ak.core_user a
            ON a.id = x.actionkit_id
        LEFT JOIN ak_phones p
            ON p.user_id = a.id
    )
    SELECT 
        ea_c.ea_first_name as first_name1, 
        ea_c.ea_last_name as last_name1,
        ea_c.ea_email as email1,
        RIGHT(REGEXP_REPLACE(ea_c.ea_phone, '[^0-9]', ''), 10) AS phone1,
        ea_c.ea_state as state1,
        ak_u.ak_first_name as first_name2, 
        ak_u.ak_last_name as last_name2,
        ak_u.ak_email as email2,
        RIGHT(REGEXP_REPLACE(ak_u.ak_phone, '[^0-9]', ''), 10) AS phone2,
        ak_u.ak_state as state2
    FROM ea_c
    INNER JOIN ak_u
        ON ea_c.rownum = ak_u.rownum
    WHERE 
        ea_c.tmc_person_id != ak_u.tmc_person_id
    ORDER BY RANDOM()
), distincts_final as (
    SELECT *, -1 as label
    FROM distincts
    LIMIT 80000
)
select *
from distincts_final