# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:26:53 2017

P03_Data Wrangling
"""

import csv
import codecs
import re
import xml.etree.cElementTree as ET
import unidecode


OSM_PATH = "P4_prague_sample.osm"

NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
LOWER = re.compile(r'^([a-z]|_)*$')

NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']

# Variables used to audit data quality
KEYS_NODE_TAGS_K = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0}
KEYS_WAY_TAGS_K = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0} 
KEYS_TAGS_K = {}

TYPES_TAGS_NODE = {}
TYPES_TAGS_WAY = {}
TYPES_NODE = {}
TYPES_WAY = {}
TYPES_WAY_NODE = {}




def remove_diacritics(string):
    pured_string = unidecode.unidecode(string)
    pured_string.encode("ascii")
    return pured_string


""" check and count formats of inputed attrib of xml element """
def audit_format(element, keys, attrib, tags):
    if element.tag == "tag" and element.get(attrib):
        if LOWER.search(element.get(attrib)):
            keys["lower"] += 1
        elif LOWER_COLON.search(element.get(attrib)):
            keys["lower_colon"] += 1
        elif PROBLEMCHARS.search(element.get(attrib)):
            keys["problemchars"] +=1
            if element.get(attrib) not in tags:
                tags[element.get(attrib)] = 0
            tags[element.get(attrib)] += 1
            #print element.get(attrib)
        else:
            keys["other"] += 1
            if element.get(attrib) not in tags:
                tags[element.get(attrib)] = 0
            tags[element.get(attrib)] += 1
            #print element.get(attrib)
    return keys


""" check element types """
def audit_type(element, fieldtypes):
    for attr in element.attrib:
        if attr not in fieldtypes:
            fieldtypes[attr] = []
        if element.get(attr) == "" or element.get(attr) == "NULL":
                if type(None) not in fieldtypes[attr]:
                    fieldtypes[attr].append(type(None))
        else:
            try:
                if type(int(element.get(attr))) not in fieldtypes[attr]:
                    fieldtypes[attr].append(type(int(element.get(attr))))
            except ValueError:
                try:
                    if type(float(element.get(attr))) not in fieldtypes[attr]:
                        fieldtypes[attr].append(type(float(element.get(attr))))
                except ValueError:
                    if type("") not in fieldtypes[attr]:
                        fieldtypes[attr].append(type(""))
    return fieldtypes


def clear_whitespace(text):
    if text.find(" "):
        text = text.strip(" ")
    return text


"""Clean and shape node or way XML element to Python dict"""
def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    
    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags = []
    tag_d = {}
    
    if element.tag == 'node':
        for attrib in NODE_FIELDS:
            if element.get(attrib):
                global TYPES_NODE
                TYPES_NODE = audit_type(element, TYPES_NODE)
                if attrib == "user":
                    node_attribs[attrib] = remove_diacritics(element.get(attrib))
                else:
                    node_attribs[attrib] = element.get(attrib)
            else:
                return
            
        for child in element:
            
            tag_d = {}
            if child.tag == "tag" and problem_chars.search(child.get("k")) == None:
                
                # auditing format and type
                global KEYS_NODE_TAGS_K
                KEYS_NODE_TAGS_K = audit_format(child, KEYS_NODE_TAGS_K, "k", KEYS_TAGS_K)
                global TYPES_TAGS_NODE
                TYPES_TAGS_NODE = audit_type(child, TYPES_TAGS_NODE)
                
                # adjusting wrong format of ";" and postcodes 
                tag_d["id"] = element.get("id")
                if child.get("v").find(";") >= 0:
                    if child.get("k") == "addr:postcode":
                        tag_d["value"] = child.get("v").replace(" ", "")
                    else:    
                        tag_d["value"] = remove_diacritics(child.get("v").replace(";", "_"))  
                else:
                    if child.get("k") == "addr:postcode":
                        tag_d["value"] = child.get("v").replace(" ", "")
                    else:
                        tag_d["value"] = remove_diacritics((child.get("v")))    
                
                # adjusting wrong format of some keys
                ind = child.get("k").find(":")
                if ind == -1:
                    tag_d["key"] = remove_diacritics(child.get("k"))
                    tag_d["type"] = "regular"
                else:
                    if child.get("k") == "uir_adr:ADRESA_KOD":
                        tag_d["key"] = "UIRA_code"
                        tag_d["type"] = "regular"
                    else:
                        temp = remove_diacritics(child.get("k")).split(":",1) 
                        tag_d["key"] = temp[1]
                        tag_d["type"] = temp[0]
                
                if tag_d:
                    tags.append(tag_d)
        
        return {'node': node_attribs, 'node_tags': tags}
    
    elif element.tag == 'way':
        for attrib in WAY_FIELDS:
            if element.get(attrib):
                global TYPES_WAY
                TYPES_WAY = audit_type(element, TYPES_WAY)
                if attrib == "user":
                    way_attribs[attrib] = remove_diacritics(element.get(attrib))
                else:
                    way_attribs[attrib] = element.get(attrib)
            else:
                return
            
        for child in element:
            tag_d = {}
            way_node = {}
            
            if child.tag == "tag" and problem_chars.search(child.get("k")) == None:
                
                # auditing format and type
                global KEYS_WAY_TAGS_K
                KEYS_WAY_TAGS_K = audit_format(child, KEYS_WAY_TAGS_K, "k", KEYS_TAGS_K)
                global TYPES_TAGS_WAY
                TYPES_TAGS_WAY = audit_type(child, TYPES_TAGS_WAY)
                 
                # adjusting wrong format of ";" and postcodes 
                tag_d["id"] = element.get("id")
                if child.get("v").find(";") >= 0:
                    if child.get("k") == "addr:postcode":
                        tag_d["value"] = child.get("v").replace(" ", "")
                    else:    
                        tag_d["value"] = remove_diacritics(child.get("v").replace(";", "_"))  
                else:
                    if child.get("k") == "addr:postcode":
                        tag_d["value"] = child.get("v").replace(" ", "")
                    else:
                        tag_d["value"] = remove_diacritics((child.get("v")))
                        
                # adjusting wrong format of some keys        
                ind = child.get("k").find(":")
                if ind == -1:
                    tag_d["key"] = remove_diacritics(child.get("k"))
                    tag_d["type"] = "regular"
                else:
                    if child.get("k") == "uir_adr:ADRESA_KOD":
                        tag_d["key"] = "UIRA_code"
                        tag_d["type"] = "regular"
                    else:
                        temp = remove_diacritics(child.get("k")).split(":",1) 
                        tag_d["key"] = temp[1]
                        tag_d["type"] = temp[0]
                if tag_d:
                    tags.append(tag_d)
                    
            elif child.tag == "nd":
                
                #audit type
                global TYPES_WAY_NODE
                TYPES_WAY_NODE = audit_type(child, TYPES_WAY_NODE)
                
                way_node["id"] = element.get("id")
                way_node["node_id"] = child.get("ref")
                way_node["position"] = len(way_nodes)
                
                if way_node:
                    way_nodes.append(way_node)
            else:
                continue
        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}
    


# Helper Functions
# ----------------------------------------------------------------------------------------------------------
def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


class UnicodeDictWriter(csv.DictWriter, object):
    """Extend csv.DictWriter to handle Unicode input"""

    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow({
            k: (v.encode('utf-8') if isinstance(v, unicode) else v) for k, v in row.iteritems()
        })

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


# Main Function
# ----------------------------------------------------------------------------------------------------------
def process_map(file_in):
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'w') as nodes_file, \
         codecs.open(NODE_TAGS_PATH, 'w') as nodes_tags_file, \
         codecs.open(WAYS_PATH, 'w') as ways_file, \
         codecs.open(WAY_NODES_PATH, 'w') as way_nodes_file, \
         codecs.open(WAY_TAGS_PATH, 'w') as way_tags_file:

        nodes_writer = UnicodeDictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = UnicodeDictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = UnicodeDictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = UnicodeDictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = UnicodeDictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()


        for element in get_element(file_in, tags=('node', 'way')):

            el = shape_element(element)

            if el:
                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])


if __name__ == '__main__':

    process_map(OSM_PATH)
    
    print "Node tags key: \n", KEYS_NODE_TAGS_K
    print "Way tags key: \n", KEYS_WAY_TAGS_K
    print "Node Tags Types: \n", TYPES_TAGS_NODE
    print "Way Tags Types: \n", TYPES_TAGS_WAY
    print "Node Types: \n", TYPES_NODE
    print "Node Types: \n", TYPES_WAY
    print "Node-Way Types: \n", TYPES_WAY_NODE
    print KEYS_TAGS_K