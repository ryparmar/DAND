# -*- coding: utf-8 -*-
"""
Created on Fri Sep 01 11:27:40 2017

@author: mrecl

P03 Data Wrangling

PY03_Quering DB
"""

import sqlite3


db = sqlite3.connect("prague.db")
c = db.cursor()


""" size of database """
""" ----------------------------------------------------------------------- """
QUERY1 =  """ PRAGMA PAGE_COUNT; """
c.execute(QUERY1)
count = c.fetchall()
QUERY2 = """ PRAGMA PAGE_SIZE; """
c.execute(QUERY2)
p_size =c.fetchall()
print "size of database:"
print (count[0][0] * p_size[0][0]) / 1000, " kilobytes"


""" number of unique users """
""" ----------------------------------------------------------------------- """
QUERY3 = """SELECT count(DISTINCT sub.uid) as unique_users 
            FROM (SELECT uid FROM nodes UNION ALL SELECT uid FROM ways) sub;"""
c.execute(QUERY3)
data = c.fetchall()
print "number of unique users: ", data[0][0]



""" number of nodes """
""" ----------------------------------------------------------------------- """
QUERY4 = """SELECT count(*) as nodes_number FROM nodes;"""
c.execute(QUERY4)
nodes = c.fetchall()
print "number of nodes: ", nodes[0][0]
  

""" number of ways """
""" ----------------------------------------------------------------------- """
QUERY5 = """ SELECT count(*) as ways_number FROM ways;"""
c.execute(QUERY5)
ways = c.fetchall()
print "number of ways: ", ways[0][0]


""" number of shops"""
""" ----------------------------------------------------------------------- """
QUERY6 = """SELECT count(key) as number_shops FROM nodes_tags
            WHERE nodes_tags.key == 'shop'; """
c.execute(QUERY6)
data = c.fetchall()
print "number of shops: ", data[0][0]


""" number of trees"""
""" ----------------------------------------------------------------------- """
QUERY7 = """SELECT count(value) as number_trees FROM nodes_tags
            WHERE nodes_tags.value == 'tree'; """
c.execute(QUERY7)
data = c.fetchall()
print "number of trees: ", data[0][0]


""" number of footways """
""" ----------------------------------------------------------------------- """
QUERY7 = """SELECT count(value) as number_footways FROM ways_tags
            WHERE ways_tags.value == 'footway'; """
c.execute(QUERY7)
data = c.fetchall()
print "number of footways: ", data[0][0]


""" share of top 10 users """
""" ----------------------------------------------------------------------- """
QUERY8 = """SELECT sub.user as unique_users, count(sub.id) as number_inputs
            FROM (SELECT user, id FROM nodes UNION ALL SELECT user, id FROM ways) sub
            GROUP BY sub.user
            ORDER BY number_inputs DESC
            LIMIT 10;"""
c.execute(QUERY8)
rows = c.fetchall()
print "top 10 contributors"
for row in rows:
    print row[0], ":", row[1]

QUERY9 = """SELECT sum(number_inputs) as top_contributors
            FROM (SELECT sub.user as unique_users, count(sub.id) as number_inputs
            FROM (SELECT user, id FROM nodes UNION ALL SELECT user, id FROM ways) sub
            GROUP BY sub.user
            ORDER BY number_inputs DESC
            LIMIT 10);"""
                             
c.execute(QUERY9)
contribution_10 = c.fetchall()
print "share of top 10 users on the total: ", round((float(contribution_10[0][0])/float((nodes[0][0]+ways[0][0]))),4)*100, "%"


""" overview amenities and art objects """
""" ----------------------------------------------------------------------- """
QUERY10 = """SELECT sub.value, count(sub.value) as number_objects
            FROM (SELECT key, value FROM nodes_tags UNION ALL SELECT key, value FROM ways_tags) as sub
            WHERE sub.key == 'amenity' or sub.key == 'artwork_type'
            GROUP BY sub.value
            ORDER BY number_objects DESC;"""
c.execute(QUERY10)
rows = c.fetchall()
for row in rows:
    print row[0], ":", row[1],","
    

QUERY11 = """SELECT sub.value, count(sub.value) as postcodes
            FROM (SELECT key, value FROM nodes_tags UNION ALL SELECT key, value FROM ways_tags) as sub 
            WHERE sub.key == 'postcode'
            GROUP BY sub.value
            ORDER BY postcodes DESC;"""
c.execute(QUERY11)
rows = c.fetchall()
print "postcodes: "
for row in rows:
    print row


db.close()