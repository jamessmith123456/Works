#-*- coding: UTF-8 -*-
# @Time    : 2019/4/12 19:53
# @Author  : xiongzongyang
# @Site    : 
# @File    : query.py
# @Software: PyCharm

from py2neo import Graph,Node,Relationship,NodeMatcher

class Query():
    def __init__(self):
        self.graph=Graph("http://localhost:7474", username="neo4j",password="123456")

    # 问题类型0，查询电影得分
    def run(self, target_question):
        print("target_question:", target_question)
        self.cql = "MATCH  (n: Architecture{question:'" + target_question + "' }) return  n.answer;"
        # find_rela  = test_graph.run("match (n:Person{name:'张学友'})-[actedin]-(m:Movie) return m.title")
        # cql = "MATCH  (n: Architecture{question: '内平开下悬窗适用于哪些窗型？' }) return  n.answer;"
        # find_answer = graph.run(cql)
        # fk = find_answer.data()
        result=[]
        find_answer = self.graph.run(self.cql)
        return list(find_answer.data()[0].values())[0]
