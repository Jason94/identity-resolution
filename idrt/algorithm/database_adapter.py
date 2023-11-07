from abc import ABC, abstractmethod
from typing import Union, Optional, Any

from pypika import Table as SQLTable
from pypika.queries import QueryBuilder

from utils import EtlTable


class DatabaseAdapter(ABC):
    """This class provides an abstract interface to the database features used by the algorithm.

    By creating your own implementation of this class, you can use the IDRT algorithm with any
    database that is capable of running standard SQL queries generated by the pypika library.

    The class has several already-implemented public methods that take a wider range of types.
    They then call an abstract, private method that takes a smaller range of types to do the
    actual work. This is usually done to make working with libraries like pypika easier
    from the call site, such as accepting pypika Table objects natively.
    """

    @abstractmethod
    def _table_exists(self, tablename: str) -> bool:
        pass

    def table_exists(self, table: Union[str, SQLTable]) -> bool:
        """Check if a given SQL table exists in the database.

        Args:
            table (Union[str, SQLTable]): Can either be a full string path to the table
                or a pypika Table object.

        Returns:
            bool: True if the table exists, False if it does not.
        """
        if isinstance(table, str):
            return self._table_exists(table)
        else:
            return self._table_exists(table.get_sql())

    @abstractmethod
    def _execute_query(self, query: str) -> Optional[EtlTable]:
        pass

    def execute_query(self, query: Union[str, QueryBuilder]) -> Optional[EtlTable]:
        """Execute a SQL query in the database.

        Args:
            query (Union[str, QueryBuilder]): A string or pypika SQL query.

        Returns:
            Optional[Table]: Data returned by the query (if any).
        """
        if isinstance(query, str):
            return self._execute_query(query)
        else:
            return self._execute_query(query.get_sql())

    @abstractmethod
    def _upsert(self, tablename: str, data: EtlTable, primary_key: Any):
        pass

    def upsert(self, table: Union[str, SQLTable], data: EtlTable, primary_key: Any):
        """Upsert data into a database table.

        Args:
            table (Union[str, SQLTable]): The table to upsert into.
            data (Table): The data to upsert.
        """
        if isinstance(table, str):
            self._upsert(table, data, primary_key)
        else:
            self._upsert(table.get_sql(), data, primary_key)

    @abstractmethod
    def _bulk_upload(self, tablename: str, data: EtlTable):
        pass

    def bulk_upload(self, table: Union[str, SQLTable], data: EtlTable):
        """Efficiently bulk upload data into the database.

        Args:
            table (Union[str, SQLTable]): Destination table.
            data (Table): Data to upload.
        """
        if isinstance(table, str):
            self._bulk_upload(table, data)
        else:
            self._bulk_upload(table.get_sql(), data)
