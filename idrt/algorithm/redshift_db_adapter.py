from typing import Optional, Any
from parsons import Table
from database_adapter import DatabaseAdapter

from parsons.databases.redshift import Redshift
from idrt.algorithm.utils import EtlTable


class RedshiftDbAdapter(DatabaseAdapter):
    def __init__(self, rs: Redshift):
        self._rs = rs

    def _table_exists(self, tablename: str) -> bool:
        return self._rs.table_exists(tablename)

    def _execute_query(self, query: str) -> Optional[EtlTable]:
        result = self._rs.query(query)
        if result is None:
            return result
        return result.to_petl()

    def _upsert(self, tablename: str, data: EtlTable, primary_key: Any):
        self._rs.upsert(Table(data), tablename, primary_key, vacuum=False)

    def _bulk_upload(self, tablename: str, data: EtlTable):
        self._rs.copy(Table(data), tablename, if_exists="drop")
