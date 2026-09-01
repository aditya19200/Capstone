// ontology_seed.cypher
//
// Mirrors backend/services/mock_db.py's ONTOLOGY dict exactly, so that
// swapping neo4j_service.py from mock to real (per its own swap
// checklist docstring) is a pure query-for-query replacement with no
// behaviour change. Node label, relationship types, and property names
// all match what neo4j_service.py's docstrings already document as the
// target Cypher:
//
//   Node label   : OntologyNode { label, description }
//   Relationships: (:OntologyNode)-[:HAS_PARENT]->(:OntologyNode)
//                  (:OntologyNode)-[:HAS_CHILD]->(:OntologyNode)
//                  (:OntologyNode)-[:RELATED_TO]->(:OntologyNode)
//
// All 10 current domains are flat (parent = None, children = []) in the
// mock, so only RELATED_TO edges are seeded below. HAS_PARENT/HAS_CHILD
// are real relationship types the moment a domain gets a parent/child —
// nothing else needs to change for that later.

CREATE CONSTRAINT ontology_node_label IF NOT EXISTS
FOR (n:OntologyNode) REQUIRE n.label IS UNIQUE;

// ---- 10 nodes (label + description copied verbatim from ONTOLOGY) ----
MERGE (n:OntologyNode {label: "Contract Law"})
SET n.description = "Governs agreements between parties, including formation, enforcement, and breach.";

MERGE (n:OntologyNode {label: "Criminal Law"})
SET n.description = "Covers offences against the state or public, prosecution, and sentencing.";

MERGE (n:OntologyNode {label: "Constitutional Law"})
SET n.description = "Deals with the interpretation and application of a country's constitution.";

MERGE (n:OntologyNode {label: "Corporate / Company Law"})
SET n.description = "Regulates the formation, governance, and dissolution of companies.";

MERGE (n:OntologyNode {label: "Property / Real Estate Law"})
SET n.description = "Covers ownership, transfer, and use of real and personal property.";

MERGE (n:OntologyNode {label: "Family Law"})
SET n.description = "Governs marriage, divorce, child custody, and related domestic matters.";

MERGE (n:OntologyNode {label: "Labour & Employment Law"})
SET n.description = "Regulates the employer-employee relationship, wages, and workplace rights.";

MERGE (n:OntologyNode {label: "Intellectual Property Law"})
SET n.description = "Covers patents, trademarks, copyrights, and trade secrets.";

MERGE (n:OntologyNode {label: "Taxation Law"})
SET n.description = "Deals with tax obligations, disputes, and compliance for individuals and entities.";

MERGE (n:OntologyNode {label: "Civil Procedure / Other"})
SET n.description = "Procedural rules for civil litigation and miscellaneous legal matters.";

// ---- RELATED_TO edges (directed, matching each node's "related" list) ----
MATCH (a:OntologyNode {label: "Contract Law"}), (b:OntologyNode {label: "Corporate / Company Law"})
MERGE (a)-[:RELATED_TO]->(b);
MATCH (a:OntologyNode {label: "Contract Law"}), (b:OntologyNode {label: "Civil Procedure / Other"})
MERGE (a)-[:RELATED_TO]->(b);

MATCH (a:OntologyNode {label: "Criminal Law"}), (b:OntologyNode {label: "Civil Procedure / Other"})
MERGE (a)-[:RELATED_TO]->(b);

MATCH (a:OntologyNode {label: "Constitutional Law"}), (b:OntologyNode {label: "Criminal Law"})
MERGE (a)-[:RELATED_TO]->(b);
MATCH (a:OntologyNode {label: "Constitutional Law"}), (b:OntologyNode {label: "Civil Procedure / Other"})
MERGE (a)-[:RELATED_TO]->(b);

MATCH (a:OntologyNode {label: "Corporate / Company Law"}), (b:OntologyNode {label: "Contract Law"})
MERGE (a)-[:RELATED_TO]->(b);
MATCH (a:OntologyNode {label: "Corporate / Company Law"}), (b:OntologyNode {label: "Taxation Law"})
MERGE (a)-[:RELATED_TO]->(b);

MATCH (a:OntologyNode {label: "Property / Real Estate Law"}), (b:OntologyNode {label: "Contract Law"})
MERGE (a)-[:RELATED_TO]->(b);
MATCH (a:OntologyNode {label: "Property / Real Estate Law"}), (b:OntologyNode {label: "Family Law"})
MERGE (a)-[:RELATED_TO]->(b);

MATCH (a:OntologyNode {label: "Family Law"}), (b:OntologyNode {label: "Property / Real Estate Law"})
MERGE (a)-[:RELATED_TO]->(b);
MATCH (a:OntologyNode {label: "Family Law"}), (b:OntologyNode {label: "Civil Procedure / Other"})
MERGE (a)-[:RELATED_TO]->(b);

MATCH (a:OntologyNode {label: "Labour & Employment Law"}), (b:OntologyNode {label: "Contract Law"})
MERGE (a)-[:RELATED_TO]->(b);
MATCH (a:OntologyNode {label: "Labour & Employment Law"}), (b:OntologyNode {label: "Civil Procedure / Other"})
MERGE (a)-[:RELATED_TO]->(b);

MATCH (a:OntologyNode {label: "Intellectual Property Law"}), (b:OntologyNode {label: "Corporate / Company Law"})
MERGE (a)-[:RELATED_TO]->(b);
MATCH (a:OntologyNode {label: "Intellectual Property Law"}), (b:OntologyNode {label: "Contract Law"})
MERGE (a)-[:RELATED_TO]->(b);

MATCH (a:OntologyNode {label: "Taxation Law"}), (b:OntologyNode {label: "Corporate / Company Law"})
MERGE (a)-[:RELATED_TO]->(b);
MATCH (a:OntologyNode {label: "Taxation Law"}), (b:OntologyNode {label: "Civil Procedure / Other"})
MERGE (a)-[:RELATED_TO]->(b);

MATCH (a:OntologyNode {label: "Civil Procedure / Other"}), (b:OntologyNode {label: "Contract Law"})
MERGE (a)-[:RELATED_TO]->(b);
MATCH (a:OntologyNode {label: "Civil Procedure / Other"}), (b:OntologyNode {label: "Family Law"})
MERGE (a)-[:RELATED_TO]->(b);
MATCH (a:OntologyNode {label: "Civil Procedure / Other"}), (b:OntologyNode {label: "Labour & Employment Law"})
MERGE (a)-[:RELATED_TO]->(b);
MATCH (a:OntologyNode {label: "Civil Procedure / Other"}), (b:OntologyNode {label: "Property / Real Estate Law"})
MERGE (a)-[:RELATED_TO]->(b);

// No HAS_PARENT / HAS_CHILD edges are seeded — every current node is
// top-level in the mock (parent: None, children: []). Add those MERGE
// statements here if/when a domain gains a real sub-category.
