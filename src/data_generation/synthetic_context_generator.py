"""
synthetic_context_generator.py

Semi-synthetic pre-transaction context generator for the dlg_gnn GraphRAG pipeline.

LABEL LEAKAGE POLICY (SCI reviewer requirement):
  - This module does NOT accept fraud labels as input.
  - Scenario type assignment is performed externally (e.g. by the calling pipeline)
    using a policy that does NOT directly map label==1 → fraud scenario.
  - Hard negatives (benign label, fraud-like language: ~35%) and
    hard positives (fraud label, benign-like language without explicit keywords: ~35%)
    are both heavily represented to prevent trivial text-based shortcuts (TASK 5.2, 5.3, 5.4).
"""

import os
import json
import random
import logging
from datetime import datetime, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Fraud scenario templates
# ---------------------------------------------------------------------------
FRAUD_TEMPLATES = {
    "investment_scam": [
        "Hey! Check out this new smart contract. It guarantees 500% daily returns! Send USDT to this pool wallet: {wallet}. Early bird bonus ends soon!",
        "Double your money in 48 hours! A registered AI trading bot is liquidating. Deposit to the broker's address: {wallet}. Verified returns.",
        "Join our VIP Telegram trading signal room. Today's coin is ready to pump. Send entry liquidity to: {wallet} and share transaction hash.",
        "Special crypto investment program offering guaranteed 25% weekly profit. Pool deposit address: {wallet}. Limited slots.",
        "Private arbitrage bot contract discovered. Forward funds to {wallet} to mirror our automated trading profits.",
    ],
    "romance_scam": [
        "Dearest, my grandfather is hospitalized and I cannot cover the emergency bill. Can you please send 1000 USDT to this wallet: {wallet}? I will pay you back.",
        "My love, I want to invest in our future home. I found a great crypto trust fund. Send your contribution to: {wallet} so we can register together.",
        "I'm stuck at customs and they require a crypto clearance deposit. Please transfer to the agent: {wallet} so I can board my flight to see you.",
        "Darling, my business account is frozen while overseas. Could you assist by sending crypto to: {wallet}? I miss you so much.",
        "Honey, I found this exclusive investment portal for couples. Let's send our joint capital to: {wallet} to start our future.",
    ],
    "phishing_url_scam": [
        "CRITICAL ALERT: Your Trust Wallet security has been compromised. Please synchronize recovery phrase at http://verification-trustwallet.com and migrate tokens to: {wallet} immediately.",
        "MetaMask Official: A new network upgrade requires immediate migration of all ERC20 tokens. Go to http://metamask-upgrade.net and migrate funds to: {wallet}.",
        "Binance Security: Unauthorized API access detected. Move your holdings to the security escrow contract at: {wallet} to prevent loss of funds.",
        "Security update notice: Re-authenticate your wallet connection at http://dapp-validator-sync.org and transfer balance to secure vault: {wallet}.",
        "Warning: Vulnerability found in old staking pool. Emergency withdraw and redeploy to audited contract: {wallet} via http://token-safe-migration.com.",
    ],
    "impersonation_scam": [
        "This is Ajay from tech support. We are auditing the department blockchain node. Please transfer the test tokens to official address: {wallet} for verification.",
        "Hello, this is Cyber Crime Division police. Your wallet is linked to darknet laundering. Move funds to court custody address: {wallet} for investigation.",
        "This is Student Union finance officer. We require all registration fees to be submitted via crypto. Please deposit to official wallet: {wallet} by tonight.",
        "Official notice from Tax Authority: Unreported staking gains detected. Pay penalty settlement to official treasury address: {wallet} to avoid legal action.",
        "Bank Compliance Dept: Suspicious inflow detected. Settle verification guarantee funds to regulatory clearing address: {wallet}.",
    ],
    "urgent_transfer_request": [
        "Hey, my car broke down on highway and towing service only accepts instant crypto transfer. Please urgently send 300 USDT to: {wallet}. Will repay tomorrow!",
        "URGENT: I'm at hospital pharmacy and need to pay for medications. Please transfer to wallet: {wallet} right now. Phone battery dying.",
        "Quick! Bidding on rare NFT auction short of gas fee. Send 0.1 ETH to: {wallet} within 5 minutes or I lose bid!",
        "Emergency: Laptop stolen while traveling and need emergency hotel funds. Transfer crypto to hostel manager: {wallet} asap.",
        "Immediate action needed: Margin call on lending position! Send collateral to: {wallet} before liquidation executes in 10 mins!",
    ],
    "fake_customer_support": [
        "Hi, Uniswap Support desk here. Your pending swap is stuck in mempool. Resolve by sending identical balance to pool validator: {wallet}.",
        "Metamask Helpdesk: To restore swap feature, authorize secondary contract validation by transferring gas fees to: {wallet}.",
        "Ledger Live support team. A firmware bug corrupted device index. Transfer funds to recovery wallet: {wallet} to secure keys.",
        "DeFi Bridge Helpdesk: Transaction hash pending relay. Please deposit validation gas fee to bridge relay agent: {wallet}.",
        "OpenSea Resolution team: Failed NFT minting claim. Pay gas fee difference to claim agent: {wallet} to unblock asset.",
    ],
    "crypto_wallet_migration_scam": [
        "Action Required: Ethereum L1 validator migration in progress. Safeguard tokens by depositing to migration wallet: {wallet} before hard fork.",
        "Important: Migrate old USDT tokens to new secure smart contract. Send old tokens to: {wallet} to receive upgraded ones.",
        "Arbitrum network upgrade: Secure bridging slots now. Send assets to bridging escrow address: {wallet} to claim native tokens.",
        "Layer 2 snapshot notice: Consolidate wallet addresses before snapshot time by transferring to registry contract: {wallet}.",
        "Token migration portal: Swap legacy ERC20 tokens for v2 governance token at contract: {wallet}.",
    ],
    "recovery_phrase_stealing_attempt": [
        "Decrypt private key offline. Input seed words at http://ledger-phrase-backup.org and back up vault to address: {wallet} to prevent loss.",
        "Free airdrop verification requires wallet confirmation. Enter recovery phrase on portal or submit verification stake to: {wallet}.",
        "Crypto backup agent: Sync hardware wallet keys to secure cloud. Transfer validation transaction to: {wallet} to verify ownership.",
        "Key recovery assistance: Backup your 12-word mnemonic phrase securely and transfer verification fee to: {wallet}.",
        "Decentralized vault recovery protocol: Confirm secret recovery phrase and initialize gas deposit to: {wallet}.",
    ],
    "high_yield_guaranteed_return_scam": [
        "Welcome to smart contract farm. Stake coins here and yield 35% APY guaranteed by smart lock. Send capital to: {wallet}.",
        "Safe Earn project: Locked staking now open. Double rewards for Polygon users. Send Matic directly to: {wallet} for auto-compounding.",
        "Private investment club fund: High yield arbitrage pool. Deposit directly to fund wallet: {wallet} for monthly payouts.",
        "Guaranteed 50% monthly profit pool launched by quant team. Send initial capital to: {wallet} before pool cap reached.",
        "Crypto wealth multiplier: Deposit and get guaranteed returns every 24 hours. Pool address: {wallet}.",
    ],
    "multi_stage_grooming_scam": [
        "Nice chatting about crypto yesterday. My mentor has inside info on new token launch. If you want in, transfer seed money to: {wallet}.",
        "Thanks for the advice on DeFi courses. Since you're interested, my group runs a private staking pool. Deposit to: {wallet} to join.",
        "Let's meet up next week. In the meantime, token price is surging. Put money in this wallet: {wallet} to catch the green candle.",
        "As we discussed over coffee, our investment circle is opening a new private liquidity node. Send participation funds to: {wallet}.",
        "I really appreciate our friendship. Let me share this private trading opportunity with you. Transfer entry stake to: {wallet}.",
    ],
}

# ---------------------------------------------------------------------------
# Hard positives: fraud transactions but WITHOUT explicit scam keywords
# (Normal everyday speech, casual transfers, ambiguous deals)
# ---------------------------------------------------------------------------
HARD_POSITIVE_TEMPLATES = [
    "Sending funds to my new account at {wallet} as discussed earlier.",
    "Processing the agreed contribution for the group project at {wallet}.",
    "Completing the transfer we talked about. Here is the address: {wallet}.",
    "Sending the amount for the deal we finalized. Wallet: {wallet}.",
    "Transferring the balance per our conversation. Recipient: {wallet}.",
    "Here is the payment for the service. Please confirm receipt at {wallet}.",
    "Settling the outstanding balance as per our agreement. Address: {wallet}.",
    "Moving funds to a new account as part of our plan: {wallet}.",
    "Hey, sending over the tokens for the lunch split: {wallet}.",
    "Payment for the freelance design work completed this week: {wallet}.",
    "Reimbursing you for the hotel booking expenses: {wallet}.",
    "Transferring budget for the office supplies order to: {wallet}.",
    "Here is my share for the shared cloud hosting subscription: {wallet}.",
    "Sending the agreed payment for the consulting hours: {wallet}.",
    "Settling the concert tickets money per our chat: {wallet}.",
    "Transferring monthly allowance to the family shared account: {wallet}.",
    "Adding liquidity to our joint shopping pool: {wallet}.",
    "Sending tokens for the birthday gift contribution to: {wallet}.",
    "Returning the borrowed funds from last weekend to: {wallet}.",
    "Regular scheduled payment for the tutoring sessions: {wallet}.",
]

# ---------------------------------------------------------------------------
# Benign templates: Everyday casual non-fraud transfers
# ---------------------------------------------------------------------------
BENIGN_TEMPLATES = [
    "Transferring pocket money to my younger sister.",
    "Paying my friend for yesterday's dinner share. It was around 25 USDT.",
    "Depositing funds to my private cold storage ledger hardware wallet.",
    "Transferring ETH from exchange wallet to my personal MetaMask wallet for gas fees.",
    "Sending contribution for the university blockchain research laboratory fund.",
    "Buying some NFTs on OpenSea marketplace.",
    "Paying registration fee for the Computer Engineering seminar.",
    "Refueling gas fees for testing smart contracts on local testnet.",
    "Sending money to my roommate for monthly electricity bill split.",
    "Withdrawing rewards from my legitimate pool staking contract.",
    "Paying for online course subscription with crypto.",
    "Contributing to the open-source developer fund for the library I use.",
    "Sending tip to a content creator via crypto.",
    "Exchanging tokens for a DeFi liquidity pool position.",
    "Paying for domain registration fees with USDC.",
    "Splitting groceries bill with my apartment flatmates.",
    "Sending monthly rent contribution to landlord wallet.",
    "Funding my secondary mobile wallet for everyday coffee payments.",
    "Purchasing a software license key using cryptocurrency.",
    "Transferring staking yields back to exchange spot account.",
]

# ---------------------------------------------------------------------------
# Hard negatives: benign transactions but with fraud-like / urgent language
# (prevents "urgent/investment/wallet → fraud" shortcut)
# ---------------------------------------------------------------------------
HARD_NEGATIVE_TEMPLATES = [
    "URGENT: Sending money to my mother for her emergency medical bill. Please process quickly to {wallet}.",
    "Transferring funds to the court escrow wallet for official apartment lease security deposit at {wallet}.",
    "Migrating my assets from old address to new address for personal safety key rotation. Address: {wallet}",
    "Sending official tuition fee to University main financial treasury contract: {wallet}.",
    "Urgently sending funds to my friend who is stranded at campus store without wallet: {wallet}.",
    "Transferring to verified escrow for the real estate transaction I am processing: {wallet}.",
    "Sending my own funds from hot wallet to cold wallet for security storage: {wallet}.",
    "Making a guaranteed stablecoin deposit to my personal savings vault: {wallet}.",
    "Urgent transfer to business partner for a legitimate time-sensitive inventory order: {wallet}.",
    "Sending identity verification deposit to government-licensed exchange: {wallet}.",
    "High priority transfer for server hosting renewal before midnight deadline: {wallet}.",
    "Official verification payment to the trademark registry escrow wallet: {wallet}.",
    "Migrating my staking positions between official Uniswap v2 and v3 liquidity pools: {wallet}.",
    "Emergency car repair bill payment to auto mechanic crypto account: {wallet}.",
    "Depositing capital into company corporate treasury multi-sig contract: {wallet}.",
    "Quick transfer needed: paying customs import tax for my international package: {wallet}.",
    "URGENT transfer: Settling hospital discharge payment for family member at {wallet}.",
    "Security migration: moving funds away from deprecated smart contract to new audited vault: {wallet}.",
    "Official legal escrow deposit for contract dispute resolution: {wallet}.",
    "Time-sensitive collateral top-up to prevent DeFi loan liquidation: {wallet}.",
]

# Prefixes and suffixes to add natural realistic linguistic variation
CONVERSATION_PREFIXES = [
    "",
    "Hi, ",
    "Hello! ",
    "FYI: ",
    "Quick note: ",
    "Per our chat, ",
    "Just confirming: ",
    "Hey there, ",
    "Regarding our earlier discussion: ",
    "Status update: ",
]

CONVERSATION_SUFFIXES = [
    "",
    " Thanks!",
    " Please confirm once received.",
    " Let me know when it clears.",
    " Have a great day.",
    " Cheers!",
    " Tx hash attached.",
    " Best regards.",
    " Will check back later.",
]


def _extract_risk_cues(text: str) -> List[str]:
    """Extract surface-level risk cues from context text (keyword-based heuristic)."""
    cues = []
    t = text.lower()
    if "guaranteed" in t or "return" in t or "yield" in t or "apy" in t or "profit" in t:
        cues.append("guaranteed return")
    if "urgent" in t or "immediately" in t or "right now" in t or "emergency" in t or "quick" in t:
        cues.append("urgent transfer")
    if "wallet" in t or "address" in t or "contract" in t:
        cues.append("external wallet request")
    if "verification" in t or "verify" in t or "confirm" in t or "identity" in t or "security" in t:
        cues.append("identity verification")
    if "escrow" in t or "treasury" in t or "custody" in t or "deposit" in t:
        cues.append("external escrow contract")
    return cues


class SyntheticContextGenerator:
    """
    Generates semi-synthetic pre-transaction contextual text for the dlg_gnn GraphRAG pipeline.

    Label Leakage Prevention:
      - Does NOT receive fraud labels as input.
      - Uses rich template pools with hard positives and hard negatives.
      - Applies natural linguistic prefixes and suffixes to defeat naive n-gram memorization.
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def generate_single(
        self,
        scenario_type: str,
        event_id: str,
        wallet_addr: str,
        tx_timestamp: datetime,
        gap_sec: int,
    ) -> dict:
        """Generate one context item for a given scenario_type (no label access)."""
        ctx_timestamp = tx_timestamp - timedelta(seconds=gap_sec)

        if scenario_type in FRAUD_TEMPLATES:
            template = self._rng.choice(FRAUD_TEMPLATES[scenario_type])
            body = template.format(wallet=wallet_addr)
        elif scenario_type == "hard_positive":
            template = self._rng.choice(HARD_POSITIVE_TEMPLATES)
            body = template.format(wallet=wallet_addr)
        elif scenario_type == "hard_negative":
            template = self._rng.choice(HARD_NEGATIVE_TEMPLATES)
            body = template.format(wallet=wallet_addr)
        else:  # benign
            body = self._rng.choice(BENIGN_TEMPLATES)

        # Add realistic variation
        prefix = self._rng.choice(CONVERSATION_PREFIXES)
        suffix = self._rng.choice(CONVERSATION_SUFFIXES)
        context_text = f"{prefix}{body}{suffix}".strip()

        risk_cues = _extract_risk_cues(context_text)

        return {
            "context_id": f"ctx_{event_id}",
            "event_id": event_id,
            "scenario_type": scenario_type,
            "context_text": context_text,
            "context_timestamp": ctx_timestamp.isoformat() + "Z",
            "transaction_timestamp": tx_timestamp.isoformat() + "Z",
            "pre_transaction_gap_sec": gap_sec,
            "risk_cues": risk_cues,
            "generation_source": "synthetic",
            "validation_status": "pending",
        }

    def generate_contexts(
        self,
        scenario_types: List[str],
        event_ids: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ) -> List[dict]:
        """Generate context records for a list of scenario_types."""
        n = len(scenario_types)
        if event_ids is None:
            event_ids = [f"tx_{i:06d}" for i in range(n)]

        base_time = datetime(2026, 1, 1, 0, 0, 0)
        records = []

        for idx, (scenario_type, event_id) in enumerate(zip(scenario_types, event_ids)):
            wallet_addr = f"0x{idx:040x}"
            tx_timestamp = base_time + timedelta(seconds=idx * 10)
            gap_sec = self._rng.randint(60, 600)
            record = self.generate_single(
                scenario_type=scenario_type,
                event_id=event_id,
                wallet_addr=wallet_addr,
                tx_timestamp=tx_timestamp,
                gap_sec=gap_sec,
            )
            records.append(record)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
            logger.info(f"Wrote {n} context records to {output_path}")

        return records


def assign_scenarios_no_leakage(
    labels,
    *,
    fraud_benign_rate: float = 0.40,
    benign_suspicious_rate: float = 0.35,
    seed: int = 42,
) -> List[str]:
    """
    Assign scenario types for a list of binary labels WITHOUT direct leakage.

    Policy (SCI-compliant):
    - Fraud nodes (label==1): 60% high-risk scenario, 40% benign/casual text (hard positive)
    - Benign nodes (label==0): 65% benign text, 35% high-risk scenario (hard negative)

    Sharing the scenario pool ensures that text alone cannot achieve trivial
    near-perfect discrimination (Acceptance criterion TASK 5.4). Real fraud detection
    requires fusing graph structural anomalies with contextual risk cues.
    """
    rng = random.Random(seed)
    fraud_scenarios = list(FRAUD_TEMPLATES.keys())
    scenario_list = []

    for label in labels:
        lbl = int(label)
        if lbl == 1:
            if rng.random() < fraud_benign_rate:
                scenario_list.append(rng.choice(["benign", "hard_positive"]))
            else:
                scenario_list.append(rng.choice(fraud_scenarios))
        else:
            if rng.random() < benign_suspicious_rate:
                scenario_list.append(rng.choice(fraud_scenarios + ["hard_negative"]))
            else:
                scenario_list.append("benign")

    return scenario_list
