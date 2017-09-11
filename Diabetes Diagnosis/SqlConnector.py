# encoding=utf-8
import pymssql


class MSSQL:
    def __init__(self, host, user, password, database, port):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port

    def __connect(self):
        self.con = pymssql.connect(self.host, self.user, self.password, self.database, self.port)
        cur = self.con.cursor()
        if not cur:
            print 'connect error!'
        else:
            return cur

    def execQury(self, sql):
        cur = self.__connect()
        cur.execute(sql)
        resultList = cur.fetchall()

        self.con.close()
        return resultList

