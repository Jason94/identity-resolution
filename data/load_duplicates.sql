SELECT 
    LOWER(ea_c.firstname) AS ea_first_name, 
    LOWER(ea_c.lastname) AS ea_last_name, 
    LOWER(ak_u.first_name) AS ak_first_name, 
    LOWER(ak_u.last_name) AS ak_last_name, 
    LOWER(mc.first_name) AS mc_first_name, 
    LOWER(mc.last_name) AS mc_last_name
FROM indivisible_infrastructure.xwalk x
LEFT JOIN indivisible_ea.contacts_iv ea_c
    ON ea_c.vanid = x.everyaction_id
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
    OR LOWER(ak_u.last_name) != LOWER(mc.last_name);
