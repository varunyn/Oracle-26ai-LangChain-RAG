"""User feedback handling for the RAG agent."""

import logging
from datetime import datetime

from ..infrastructure.db_utils import get_pooled_connection

logger = logging.getLogger(__name__)


class RagFeedback:
    """
    To register user feedback in the RAG_FEEDBACK table.
    """

    def __init__(self):
        """
        Init
        """

    def table_exists(self, table_name: str) -> bool:
        """
        Check that the table exist in the current schema
        """
        sql = """
            SELECT COUNT(*)
            FROM user_tables
            WHERE table_name = :tn
        """
        with get_pooled_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, tn=table_name.upper())
                row = cursor.fetchone()
                count = int(row[0]) if row else 0
                return count > 0

    def _create_table(self):
        """
        Create the table if it doesn't exists
        """
        logger.info("Creating table RAG_FEEDBACK...")

        ddl_instr = """
        CREATE TABLE RAG_FEEDBACK (
        ID         NUMBER GENERATED ALWAYS AS IDENTITY
                    (START WITH 1 INCREMENT BY 1) PRIMARY KEY,
        CREATED_AT DATE   DEFAULT SYSDATE        NOT NULL,
        QUESTION   CLOB                          NOT NULL,
        ANSWER     CLOB                          NOT NULL,
        FEEDBACK   NUMBER(2,0)                   NOT NULL
        )
        """
        with get_pooled_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl_instr)

    def insert_feedback(self, question: str, answer: str, feedback: int):
        """Insert a new feedback record into RAG_FEEDBACK table."""
        if feedback < 1 or feedback > 5:
            raise ValueError("Feedback must be a number between 1 and 5.")

        sql = """
            INSERT INTO RAG_FEEDBACK (CREATED_AT, QUESTION, ANSWER, FEEDBACK)
            VALUES (:created_at, :question, :answer, :feedback)
        """

        if not self.table_exists("RAG_FEEDBACK"):
            # table doesn't exists
            logger.info("Table RAG_FEEDBACK doesn't exist...")
            self._create_table()

        with get_pooled_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                sql,
                {
                    "created_at": datetime.now(),
                    "question": question,
                    "answer": answer,
                    "feedback": feedback,
                },
            )
            conn.commit()
            cursor.close()
