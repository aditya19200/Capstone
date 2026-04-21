"""
Hard evaluation: tests model on cross-domain, ambiguous, and edge-case inputs
that probe known weaknesses identified during training.
"""
import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict

MODEL_PATH = os.getenv("MODEL_PATH", "models/v1")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
LABEL_MAP = {int(k): v for k, v in model.config.id2label.items()}

# (text, expected_label_id, difficulty_category, why_hard)
HARD_TESTS = [
    # === CROSS-DOMAIN BLEED ===
    # Criminal disputes that LOOK like other domains
    ("The accused, a director of the company, was charged under Section 420 IPC for cheating the bank by submitting forged financial statements to obtain a loan of Rs. 50 crores. The Sessions Court rejected the bail application citing the magnitude of the fraud.",
     1, "cross_domain", "Criminal but mentions company, director, bank, loan — looks Corporate/Contract"),

    ("During the family partition suit, allegations surfaced that the elder brother had forged the father's will. The complainant filed an FIR under Sections 467 and 471 IPC for forgery of valuable security. The trial is now pending before the Magistrate.",
     1, "cross_domain", "Criminal forgery but in family/property context"),

    ("The dowry harassment case under Section 498A IPC was filed by the wife alleging that her husband and in-laws demanded an additional Rs. 10 lakhs and a flat in Mumbai as part of the matrimonial settlement. The High Court denied anticipatory bail to the in-laws.",
     1, "cross_domain", "Criminal but heavy family/property vocabulary"),

    # Contract disputes that LOOK like other domains
    ("The cloud service provider was sued for breach of the Service Level Agreement after their data centre outage caused a 72-hour downtime for the e-commerce platform. The plaintiff seeks liquidated damages of Rs. 5 crores under Clause 12 of the SLA.",
     0, "cross_domain", "Contract dispute but looks corporate/IT"),

    ("The exclusive distributorship agreement between the parties contained a non-compete clause restricting the distributor from selling competing products for two years post-termination. The High Court held the restriction unreasonable and unenforceable as it amounted to restraint of trade.",
     0, "cross_domain", "Contract law but mentions IP-like terms"),

    # Property disputes that LOOK criminal/civil
    ("The plaintiff alleges that the defendant fraudulently obtained the sale deed of the ancestral property by impersonating the original owner. The civil suit seeks declaration that the sale deed is void ab initio and a permanent injunction restraining the defendant from selling the property.",
     4, "cross_domain", "Property dispute but mentions fraud/criminal language"),

    ("The lease deed executed for 99 years grants the lessee the right to construct a multi-storey commercial complex on the demised premises. The dispute concerns the lessor's claim that the tenant violated the building plan approved by the municipal authority.",
     4, "cross_domain", "Property/lease but corporate flavor"),

    # Corporate that LOOKS like Contract/Tax
    ("The minority shareholders filed a derivative suit alleging that the directors wrongfully diverted Rs. 200 crores to a related party at non-arm's length terms. The petition under Sections 241-242 of the Companies Act seeks removal of the board and restitution of the diverted funds.",
     3, "cross_domain", "Corporate but heavy contract/transaction vocabulary"),

    ("The NCLT admitted the Section 7 IBC petition filed by the financial creditor for default of Rs. 100 crores. The interim resolution professional was appointed and a moratorium under Section 14 was imposed. The Committee of Creditors will evaluate the resolution plan submitted by three prospective acquirers.",
     3, "cross_domain", "Insolvency/corporate but heavy financial/contract terms"),

    # IP that LOOKS like Contract/Corporate
    ("Following the termination of the Master Franchise Agreement, the franchisor filed a suit seeking a permanent injunction restraining the former franchisee from continuing to use the registered trademark, the proprietary store layout, and the trade dress at its retail outlets.",
     7, "cross_domain", "IP infringement but heavy contract/franchise terms"),

    ("The acquirer in the share purchase transaction discovered that the target company's flagship software incorporated open-source components subject to a strict copyleft license. The buyer sued for breach of the IP warranties in the SPA, seeking indemnification of Rs. 25 crores.",
     7, "cross_domain", "IP issue but in M&A/contract context"),

    # Tax that LOOKS like Corporate
    ("The transfer pricing officer made an upward adjustment of Rs. 80 crores to the international transactions of the Indian subsidiary, rejecting the comparable companies selected by the assessee. The ITAT directed the TPO to exclude functionally dissimilar entities and recompute the arm's length margin.",
     8, "cross_domain", "Tax but corporate/transactional"),

    ("The amalgamation of the loss-making subsidiary into the holding company was challenged by the income tax department, which alleged that the scheme was designed solely to set off accumulated losses. The Tribunal upheld the scheme citing commercial substance under Section 72A.",
     8, "cross_domain", "Tax issue arising from corporate restructuring"),

    # Family that LOOKS criminal
    ("The wife filed a domestic violence complaint under the PWDV Act 2005, seeking a residence order in the matrimonial home and monthly maintenance of Rs. 50,000. The Magistrate also restrained the husband from alienating the joint property pending disposal of the petition.",
     5, "cross_domain", "Family law but mentions criminal procedure & property"),

    # Labour that LOOKS like Contract
    ("The IT employee was terminated without notice for breach of the non-compete clause in his employment contract after he joined a competing firm. The Labour Court held the termination invalid for failure to follow the principles of natural justice and ordered reinstatement with back wages.",
     6, "cross_domain", "Labour but heavy contract terminology"),

    # Constitutional that LOOKS like Civil Procedure
    ("The petitioner invoked Article 226 of the Constitution to challenge the order of the disciplinary authority dismissing him from service for alleged misconduct. The High Court held that the inquiry violated the fundamental right to a fair hearing under Article 14.",
     2, "cross_domain", "Constitutional but procedural appearance"),

    # Civil Procedure pure
    ("The plaint was returned by the trial court for presentation before the proper court having pecuniary jurisdiction. The plaintiff filed an appeal under Order XLIII Rule 1 of the CPC challenging the jurisdictional finding. The High Court set aside the order observing that valuation must be based on the relief claimed.",
     9, "cross_domain", "Pure procedural — easy to mistake for substantive"),

    # === LONG REAL-STYLE INPUTS ===
    ("In the matter between ABC Pharmaceuticals Ltd and the Controller of Patents, the appellant challenged the rejection of its patent application for a novel formulation of an existing molecule. The Patent Office had invoked Section 3(d) of the Patents Act, 1970 holding that mere discovery of a new form of a known substance which does not result in enhancement of known efficacy is not patentable. The appellant argued that the new polymorphic form demonstrated 40% improved bioavailability supported by clinical data submitted as Annexure A. The Controller rejected the efficacy data citing absence of comparative therapeutic outcome studies. On appeal, the IPAB remanded the matter for fresh consideration directing the Controller to consider whether bioavailability enhancement could constitute therapeutic efficacy in the specific context of the molecule.",
     7, "long_realistic", "Realistic IP case length and complexity"),

    ("The petitioner is a daily wage worker who was engaged by the State Electricity Board for over 12 years performing duties identical to those of regularised employees. Despite repeated representations and a circular issued by the Department of Personnel directing regularisation of similarly situated workers, the petitioner was denied regularisation on the ground that he had not been engaged through the prescribed selection process. The Industrial Tribunal had earlier held that the engagement was not casual but quasi-permanent. The State invoked the Constitution Bench ruling in Umadevi to argue that no right to regularisation exists for irregularly appointed workers.",
     6, "long_realistic", "Long labour case mixing constitutional precedent"),

    # === GENUINELY AMBIGUOUS / EDGE ===
    ("The contract for the sale of immovable property included an arbitration clause requiring all disputes to be referred to a sole arbitrator. When a dispute arose regarding the title to the property, the seller invoked arbitration while the buyer filed a civil suit for specific performance. The Supreme Court held that questions of title to immovable property are arbitrable.",
     0, "ambiguous", "Genuinely overlaps Contract/Property/Civil Procedure"),

    ("The petitioner challenged the GST notification levying tax on royalty payments to the State Government for mining leases, contending that royalty is in the nature of tax and not consideration for services. The Supreme Court referred the matter to a larger bench observing the conflict with earlier rulings.",
     8, "ambiguous", "Tax+Constitutional+Property — genuinely contested"),

    # === OUT-OF-DISTRIBUTION / NON-LEGAL ===
    ("The novel reactive programming framework leverages observable streams to handle asynchronous data flows in modern web applications. Developers can compose complex event pipelines using operators like map, filter, and merge.",
     -1, "ood_nonlegal", "Pure software engineering — model should be uncertain"),

    ("The patient presented with acute chest pain radiating to the left arm, accompanied by diaphoresis and shortness of breath. The ECG showed ST-elevation in leads V1 to V4 confirming an anterior wall myocardial infarction. Immediate angioplasty was performed.",
     -1, "ood_nonlegal", "Pure medical content — model should be uncertain"),

    ("The recipe calls for two cups of all-purpose flour, one teaspoon of baking soda, and a half cup of unsalted butter. Mix the dry ingredients separately before folding them into the wet mixture. Bake at 180 degrees Celsius for 25 minutes.",
     -1, "ood_nonlegal", "Cooking recipe — model should be uncertain"),

    ("The Constitutional position regarding free speech in academic settings has been the subject of ongoing scholarly debate, particularly following recent campus controversies. Some scholars argue for absolute protection while others advocate balancing tests.",
     2, "ambiguous", "Discusses Constitutional law but as commentary, not a case"),

    ("This article examines the historical evolution of contract law principles from Roman antiquity through the development of the common law tradition in medieval England.",
     0, "ambiguous", "Academic discussion of contract law, not a dispute"),
]


def predict_one(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    conf, pid = torch.max(probs, dim=-1)
    return pid.item(), conf.item(), probs.cpu().tolist()


# Run evaluation
print(f"\n{'='*100}")
print(f"HARD EVALUATION — {len(HARD_TESTS)} cases")
print(f"{'='*100}\n")

results = []
by_category = defaultdict(lambda: {"correct": 0, "total": 0, "uncertain_count": 0})
LOW_CONF_THRESHOLD = 0.5  # if model is uncertain, count separately

for text, expected, category, why in HARD_TESTS:
    pred_id, conf, all_probs = predict_one(text)
    pred_label = LABEL_MAP[pred_id]
    
    # OOD cases (expected = -1) are correct if model has LOW confidence
    if expected == -1:
        is_correct = conf < LOW_CONF_THRESHOLD
        is_uncertain = conf < LOW_CONF_THRESHOLD
        expected_str = "(non-legal — should be uncertain)"
    else:
        is_correct = pred_id == expected
        is_uncertain = conf < LOW_CONF_THRESHOLD
        expected_str = LABEL_MAP[expected]
    
    by_category[category]["total"] += 1
    if is_correct:
        by_category[category]["correct"] += 1
    if is_uncertain:
        by_category[category]["uncertain_count"] += 1
    
    mark = "✓" if is_correct else "✗"
    conf_marker = " [LOW CONF]" if is_uncertain else ""
    print(f"{mark} [{category:15s}] Expected: {expected_str:35s}")
    print(f"   Predicted: {pred_label:30s} (conf: {conf:.3f}){conf_marker}")
    print(f"   Why hard:  {why}")
    print(f"   Text:      {text[:120]}...")
    print()
    
    results.append({
        "text": text[:200],
        "expected": expected_str,
        "predicted": pred_label,
        "confidence": round(conf, 4),
        "category": category,
        "is_correct": is_correct,
    })

# Summary
print(f"\n{'='*100}")
print("SUMMARY BY CATEGORY")
print(f"{'='*100}")
total_correct = 0
total = 0
for cat, stats in by_category.items():
    total_correct += stats["correct"]
    total += stats["total"]
    pct = 100 * stats["correct"] / stats["total"]
    print(f"  {cat:18s}  {stats['correct']:2d} / {stats['total']:2d}  ({pct:.0f}%)   uncertain: {stats['uncertain_count']}")
print(f"\n  {'TOTAL':18s}  {total_correct:2d} / {total:2d}  ({100*total_correct/total:.0f}%)")

# Save detailed results
os.makedirs("metrics", exist_ok=True)
with open("metrics/v1_hard_test.json", "w") as f:
    json.dump({
        "total": total,
        "correct": total_correct,
        "accuracy": round(total_correct / total, 4),
        "by_category": dict(by_category),
        "results": results,
    }, f, indent=2)
print(f"\nDetailed results saved to metrics/v1_hard_test.json")
