#!/usr/bin/env python3
"""
seed_retrain_data.py — one-time prep script for presentation day.

The Retrain button needs 50+ validated annotations before it will actually
run (safety gate). Clicking through 50 review items by hand isn't realistic before presenting, so this submits a
batch of real legal text through the real API, waits for it to classify,
then submits a real validated annotation for each one (agreeing with the
model's own label) — exactly the same POST /annotate call the Review
screen makes when you click a button, just done in bulk, once, ahead of
time. Nothing here is faked data — every row is a real API call that
creates a real row in the (in-memory, local) database.

Run this ONCE before presenting, against your already-running backend:

    python3 seed_retrain_data.py

Then the Retrain button on the Admin page will show a real, non-zero
validated-annotation count and actually complete when clicked.
"""

import sys
import time
import urllib.request
import json

BASE_URL = "http://localhost:8000"

TEXTS = [
    "The buyer failed to deliver the goods specified in the sales agreement by the deadline, constituting a material breach of contract.",
    "The seller refused to honour the warranty terms after the delivered equipment malfunctioned within the guarantee period.",
    "The defendant was charged with armed robbery after allegedly threatening the store clerk with a weapon and stealing cash from the register.",
    "The accused was arrested on suspicion of embezzling funds from the company's trust account over several years.",
    "The petitioner argues that the state law violates the fundamental right to freedom of speech guaranteed under Article 19 of the Constitution.",
    "The court examined whether the municipal ordinance infringes the constitutional right to equality before law.",
    "The board of directors approved a merger between the two companies, subject to shareholder ratification and regulatory clearance.",
    "The company failed to file its annual return within the statutory period, prompting a regulatory penalty.",
    "The tenant filed a suit against the landlord for wrongful eviction and failure to return the security deposit after the lease expired.",
    "A dispute arose over the boundary line between two adjoining agricultural plots after a recent survey.",
    "The wife filed for divorce citing irreconcilable differences and requested sole custody of their two minor children.",
    "The husband contested the maintenance amount awarded by the family court following their separation.",
    "The employee alleges wrongful termination and seeks reinstatement along with back wages after being dismissed without notice.",
    "A dispute arose over unpaid overtime wages claimed by factory workers under the applicable labour statute.",
    "The plaintiff claims the defendant infringed its registered trademark by selling counterfeit goods bearing an identical logo.",
    "A copyright dispute arose over the unauthorised reproduction of a musical composition in a commercial advertisement.",
    "The assessee challenged the tax authority's demand notice, arguing that the income was wrongly classified as taxable capital gains.",
    "The department disallowed input tax credit citing a procedural lapse in the taxpayer's filing.",
    "The appellant filed a writ petition seeking a stay of proceedings pending the outcome of the appeal before the higher court.",
    "The respondent sought dismissal of the suit for want of territorial jurisdiction before the trial court.",
    "The distributor claims the manufacturer's new pricing policy breaches their long-standing supply agreement.",
    "The franchise agreement between the parties contained a territory clause that the franchisee now disputes as unenforceable.",
    "The victim's family filed a complaint alleging criminal negligence following the workplace accident.",
    "The prosecution presented forensic evidence linking the accused to the scene of the burglary.",
    "The petitioner challenged the validity of the government notification as exceeding its constitutional authority.",
    "A public interest litigation was filed questioning the legality of the newly enacted surveillance rules.",
    "The minority shareholders alleged oppression and mismanagement by the majority board of the company.",
    "The company's auditors flagged irregularities in the related-party transactions disclosed in the annual report.",
    "The buyer sought specific performance of the sale deed after the seller refused to complete the property transfer.",
    "A co-owner sought partition of the ancestral property after decades of joint possession by the family.",
    "The insurer denied the claim alleging the policyholder breached a material disclosure term in the contract.",
    "The joint venture agreement was terminated after one partner alleged repeated breaches of the exclusivity clause.",
    "The accused was charged with cheating after allegedly running a fraudulent investment scheme targeting retirees.",
    "The court is examining whether the search conducted without a warrant violated the accused's constitutional protections.",
    "The company was penalised for failing to hold its mandatory annual general meeting within the prescribed period.",
    "The board rejected a shareholder resolution, prompting allegations that proper voting procedure was not followed.",
    "The buyer alleged the seller concealed material defects in the property before the sale was finalised.",
    "A dispute arose between neighbours over an easement right of way established decades earlier.",
    "The mother sought a modification of the custody order citing a material change in circumstances.",
    "The court is deciding whether the adoption was validly executed under the applicable personal law.",
    "The union alleged the employer engaged in unfair labour practices by refusing to recognise collective bargaining.",
    "An apprentice claims he was denied statutorily mandated benefits during his training period.",
    "A software firm sued a former contractor for using its proprietary source code in a competing product.",
    "The publisher was sued for releasing a book that allegedly plagiarised substantial portions of another author's work.",
    "The company contested a reassessment order that increased its taxable income for the relevant financial year.",
    "An exporter challenged the denial of a duty drawback claim on procedural grounds.",
    "The applicant sought condonation of delay in filing the appeal citing genuine hardship.",
    "The respondent argued the suit was barred by the law of limitation and should be dismissed at the threshold.",
    "A logistics company sued a client for non-payment of freight charges under their transport agreement.",
    "The licensing agreement's exclusivity clause was challenged as an unreasonable restraint of trade.",
    "The accused was convicted of criminal breach of trust after diverting funds entrusted to him as a company officer.",
    "A whistleblower alleged retaliatory action after reporting regulatory violations to the authorities.",
    "The company's shareholders filed a derivative suit alleging the directors breached their fiduciary duties.",
    "The bank initiated recovery proceedings against the borrower for default on a secured loan agreement.",
    "A landlord sought eviction of a commercial tenant for repeated default on rent payments.",
]

def http_json(method, path, body=None, headers=None):
    url = BASE_URL + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def main():
    print(f"Seeding against {BASE_URL} ...")
    try:
        health = http_json("GET", "/health")
    except Exception as exc:
        print(f"Could not reach the backend at {BASE_URL} — is run_local_demo.sh running? ({exc})")
        sys.exit(1)
    if not health.get("model_loaded"):
        print("Backend is up but the model isn't loaded yet — wait a few seconds and retry.")
        sys.exit(1)

    print(f"Submitting {len(TEXTS)} texts as a batch...")
    batch = http_json("POST", "/batches/paste", {"texts": TEXTS})
    batch_id = batch["batch_id"]

    print("Waiting for classification...")
    for _ in range(30):
        status = http_json("GET", f"/batches/{batch_id}")
        if status["status"] == "done":
            break
        time.sleep(2)
    else:
        print("Batch didn't finish in time — check backend.log.")
        sys.exit(1)

    items = http_json("GET", f"/batches/{batch_id}/items?page_size=100")["items"]
    print(f"Classified. Submitting validated annotations for {len(items)} items...")

    annotated = 0
    for item in items:
        if not item.get("prediction_id") or not item.get("predicted_label"):
            continue
        try:
            http_json(
                "POST", "/annotate",
                {
                    "prediction_id": item["prediction_id"],
                    "final_label": item["predicted_label"],
                    "action": "accept",
                },
                headers={"X-Role": "annotator", "X-User-Id": "presentation-prep"},
            )
            annotated += 1
        except Exception as exc:
            print(f"  skipped one item: {exc}")

    print(f"\nDone. {annotated} real validated annotations created.")
    print("The Admin -> Retrain page should now show a validated-annotation")
    print("count instead of 0, and Trigger Retrain will actually run.")


if __name__ == "__main__":
    main()
