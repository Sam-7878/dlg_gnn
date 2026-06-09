import os
import json
import random
import torch
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 10 Fraud scenarios templates
FRAUD_TEMPLATES = {
    "investment_scam": [
        "Hey! Check out this new smart contract. It guarantees 500% daily returns! Send USDT to this pool wallet to secure your spot: {wallet}. Early bird bonus ends soon!",
        "Double your money in 48 hours! A registered AI trading bot is liquidating. Deposit to the broker's address: {wallet}. Verified returns.",
        "Join our VIP Telegram trading signal room. Today's coin is ready to pump. Send entry liquidity to: {wallet} and share transaction hash."
    ],
    "romance_scam": [
        "Dearest, my grandfather is hospitalized and I cannot cover the emergency bill. Can you please send 1000 USDT to this wallet address: {wallet}? I will pay you back as soon as I return.",
        "My love, I want to invest in our future home. I found a great crypto trust fund. Send your contribution to: {wallet} so we can register together.",
        "I'm stuck at customs and they require a crypto clearance deposit. Please transfer to the agent: {wallet} so I can board my flight to see you."
    ],
    "phishing_url_scam": [
        "CRITICAL ALERT: Your Trust Wallet security has been compromised. Please synchronize your recovery phrase by visiting http://verification-trustwallet.com and migrate your tokens to: {wallet} immediately.",
        "MetaMask Official: A new network upgrade requires immediate migration of all ERC20 tokens. Go to http://metamask-upgrade.net and migrate your funds to: {wallet}.",
        "Binance Security: Unauthorized API access detected. Move your holdings to the security escrow contract at: {wallet} to prevent loss of funds."
    ],
    "impersonation_scam": [
        "This is Ajay from ajou university tech support. We are auditing the department blockchain node. Please transfer the test tokens to Ajou official address: {wallet} for verification.",
        "Hello, this is Ajou police department Cyber Crime Division. Your wallet is linked to a darknet money laundering case. Move funds to the court custody address: {wallet} for investigation.",
        "This is AJOU Student Union finance officer. We require all registration fees to be submitted via crypto. Please deposit to our official wallet: {wallet} by tonight."
    ],
    "urgent_transfer_request": [
        "Hey, my car broke down on the highway and the towing service only accepts instant crypto transfer. Please urgently send 300 USDT to: {wallet}. Will repay tomorrow!",
        "URGENT: I'm at the hospital pharmacy and need to pay for medications. Please transfer to my wallet: {wallet} right now. Phone battery is dying.",
        "Quick! I am bidding on a rare NFT auctions and I'm short of gas fee. Send 0.1 ETH to: {wallet} within 5 minutes or I will lose the bid!"
    ],
    "fake_customer_support": [
        "Hi, this is Uniswap Support desk. Your pending transaction is stuck in the mempool. Please resolve it by sending the identical balance to our pool validator: {wallet}.",
        "Metamask Helpdesk: To restore your swap feature, authorize the secondary smart contract validation by transferring gas fees to: {wallet}.",
        "Ledger Live support team. A firmware bug has corrupted your device index. Transfer your funds to our recovery wallet: {wallet} to secure your keys."
    ],
    "crypto_wallet_migration_scam": [
        "Action Required: Ethereum L1 validator migration in progress. Safeguard your tokens by depositing to our designated migration wallet: {wallet} before the hard fork.",
        "Important: Migrate your old USDT tokens to the new secure smart contract. Send your old tokens to: {wallet} and you will receive the upgraded ones.",
        "Arbitrum network upgrade: Secure your bridging slots now. Send assets to the bridging escrow address: {wallet} to claim L2 native tokens."
    ],
    "recovery_phrase_stealing_attempt": [
        "Decrypt your private key offline. Input your seed words at http://ledger-phrase-backup.org and back up your vault to address: {wallet} to prevent recovery failure.",
        "Ajou Blockchain Club: Free aidrop verification requires wallet confirmation. Enter your recovery phrase on our portal or submit verification stake to: {wallet}.",
        "Crypto backup agent: Sync your hardware wallet keys to our secure cloud. Transfer a validation transaction to: {wallet} to verify ownership."
    ],
    "high_yield_guaranteed_return_scam": [
        "Welcome to the smart contract farm. Stake your coins here and yield 35% APY guaranteed by smart lock. Send capital to: {wallet}.",
        "Safe Earn project: Locked staking is now open. Double rewards for Polygon users. Send Matic directly to: {wallet} to activate auto-compounding.",
        "Ajou Investment Club private fund: High yield arbitrage pool. Deposit directly to Ajou student fund wallet: {wallet} for monthly payouts."
    ],
    "multi_stage_grooming_scam": [
        "It was nice chatting about crypto utility yesterday. My uncle has inside information on a new token launch. If you want to join, transfer seed money to: {wallet}.",
        "Thanks for the advice on Ajou courses. Since you are interested in DeFi, my group is running a private staking pool. Deposit to: {wallet} to join us.",
        "Let's meet up at Ajou campus next week. In the meantime, the token price is surging. Put your money in this wallet: {wallet} to catch the green candle."
    ]
}

# Benign scenario templates
BENIGN_TEMPLATES = [
    "Transferring pocket money to my younger sister.",
    "Paying my friend for yesterday's dinner share. It was around 25 USDT.",
    "Depositing funds to my private cold storage ledger hardware wallet.",
    "Transferring ETH from exchange wallet to my personal MetaMask wallet for gas fees.",
    "Sending contribution for the Ajou blockchain research project laboratory fund.",
    "Buying some NFTs on OpenSea marketplace.",
    "Paying registration fee for the Ajou Computer Engineering seminar.",
    "Refueling gas fees for testing smart contracts on local testnet.",
    "Sending money to my roommate for monthly electricity bill split.",
    "Withdrawing rewards from my legitimate pool staking contract."
]

# Hard negative templates (benign but triggers urgency/escrow keywords)
HARD_NEGATIVE_TEMPLATES = [
    "URGENT: Sending money to my mother for her emergency medical bill. Please process quickly.",
    "Transferring funds to the court escrow wallet for my official apartment lease security deposit.",
    "Migrating my assets from old address to new address for personal safety rotation. Address: {wallet}",
    "Sending official tuition fee to Ajou University main financial treasury contract address: {wallet}.",
    "Urgently sending funds to my friend who is stranded at the Ajou campus store with no wallet."
]

class SyntheticContextGenerator:
    """
    Generates synthetic scam or benign contexts corresponding to GoG transaction labels.
    """
    def __init__(self, seed: int = 42):
        random.seed(seed)
        
    def generate_contexts(self, labels: torch.Tensor, output_path: str, max_nodes: int = 5000):
        """
        Generates context logs matching node labels.
        """
        logger.info(f"Generating synthetic contexts based on {len(labels)} labels. Max nodes: {max_nodes}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        num_generate = min(max_nodes, len(labels))
        
        base_time = datetime(2026, 1, 1, 0, 0, 0)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for idx in range(num_generate):
                label = int(labels[idx].item())
                event_id = f"tx_{idx:06d}"
                user_id = f"user_{idx:05d}"
                wallet_addr = f"0x{idx:040x}"  # Synthesize wallet address
                
                # Setup timestamps
                tx_timestamp = base_time + timedelta(seconds=idx * 10)
                # Context must happen pre-transaction (e.g. 1 to 10 minutes earlier)
                gap_sec = random.randint(60, 600)
                ctx_timestamp = tx_timestamp - timedelta(seconds=gap_sec)
                
                risk_cues = []
                
                if label == 1:
                    # Select random fraud scenario
                    scenario = random.choice(list(FRAUD_TEMPLATES.keys()))
                    template = random.choice(FRAUD_TEMPLATES[scenario])
                    context_text = template.format(wallet=wallet_addr)
                    
                    # Extract typical cues
                    if "guarantee" in context_text.lower() or "return" in context_text.lower():
                        risk_cues.append("guaranteed return")
                    if "urgent" in context_text.lower() or "immediately" in context_text.lower():
                        risk_cues.append("urgent transfer")
                    if "wallet" in context_text.lower() or "address" in context_text.lower():
                        risk_cues.append("external wallet request")
                    if "verification" in context_text.lower() or "verify" in context_text.lower():
                        risk_cues.append("identity verification")
                else:
                    # Regular or Hard Negative
                    is_hard_neg = random.random() < 0.15
                    if is_hard_neg:
                        scenario = "hard_negative"
                        template = random.choice(HARD_NEGATIVE_TEMPLATES)
                        context_text = template.format(wallet=wallet_addr)
                        # Hard negative cues that mimic risk indicators
                        if "urgent" in context_text.lower():
                            risk_cues.append("urgent transfer")
                        if "escrow" in context_text.lower() or "treasury" in context_text.lower():
                            risk_cues.append("external escrow contract")
                    else:
                        scenario = "benign"
                        context_text = random.choice(BENIGN_TEMPLATES)
                
                # Construct JSON item
                item = {
                    "context_id": f"ctx_{idx:06d}",
                    "event_id": event_id,
                    "user_id": user_id,
                    "label": label,
                    "scenario_type": scenario,
                    "context_text": context_text,
                    "context_timestamp": ctx_timestamp.isoformat() + "Z",
                    "transaction_timestamp": tx_timestamp.isoformat() + "Z",
                    "pre_transaction_gap_sec": gap_sec,
                    "risk_cues": risk_cues,
                    "generation_source": "synthetic",
                    "validation_status": "pending"
                }
                
                f.write(json.dumps(item) + "\n")
                
        logger.info(f"Successfully wrote {num_generate} synthetic contexts to {output_path}")

def main():
    # Fallback simulation if GoG graph is not loaded yet
    # Try to load labels from standard GoG dataset path
    gog_path = "D:\\_Work\\_data\\GoG\\polygon\\polygon_hybrid_graph.pt"
    if not os.path.exists(gog_path):
        gog_path = "/mnt/d/_Work/_data/GoG/polygon/polygon_hybrid_graph.pt"
        
    labels = None
    if os.path.exists(gog_path):
        try:
            data_dict = torch.load(gog_path)
            labels = data_dict['labels']
            logger.info("Loaded actual GoG labels.")
        except Exception as e:
            logger.error(f"Failed to load GoG graph: {e}")
            
    if labels is None:
        # Create virtual labels for testing purposes
        logger.info("GoG graph not found, synthesizing 10,000 virtual labels...")
        labels = torch.zeros(10000, dtype=torch.long)
        # Inject 5% outliers
        anom_idx = torch.randperm(10000)[:500]
        labels[anom_idx] = 1
        
    gen = SyntheticContextGenerator(seed=42)
    output = "d:\\_Work\\goat_bank\\dlg_gnn\\data\\contexts\\synthetic_contexts.jsonl"
    gen.generate_contexts(labels, output, max_nodes=5000)

if __name__ == "__main__":
    main()
