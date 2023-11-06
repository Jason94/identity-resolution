from typing import Optional, Any
from parsons import Table
from database_adapter import DatabaseAdapter

from parsons.databases.redshift import Redshift


class RedshiftDbAdapter(DatabaseAdapter):
    def __init__(self, rs: Redshift):
        self._rs = rs

    def _table_exists(self, tablename: str) -> bool:
        return self._rs.table_exists(tablename)

    def _execute_query(self, query: str) -> Optional[Table]:
        return self._rs.query(query)

    def _upsert(self, tablename: str, data: Table, primary_key: Any):
        self._rs.upsert(data, tablename, primary_key, vacuum=False)

    def _bulk_upload(self, tablename: str, data: Table):
        self._rs.copy(data, tablename, if_exists="drop")
