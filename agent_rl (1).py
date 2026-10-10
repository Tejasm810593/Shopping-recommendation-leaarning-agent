import csv
import re
import math
import json
import os
import random
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime


# =============================================================================
# COLUMN MAPPER
# =============================================================================

def build_col_map(fieldnames):
    normalized = {f.strip().lower().replace(" ", "_").replace("-", "_"): f
                  for f in fieldnames}

    def find(candidates):
        for c in candidates:
            if c in normalized:
                return normalized[c]
        return None

    return {
        "product_id":          find(["product_id", "id", "asin", "pid"]),
        "product_name":        find(["product_name", "name", "title"]),
        "category":            find(["category", "categories", "category_name"]),
        "discounted_price":    find(["discounted_price", "sale_price", "price"]),
        "actual_price":        find(["actual_price", "original_price", "mrp"]),
        "discount_percentage": find(["discount_percentage", "discount_percent", "discount"]),
        "rating":              find(["rating", "ratings", "avg_rating", "stars"]),
        "rating_count":        find(["rating_count", "ratings_count", "num_ratings", "no_of_ratings"]),
        "about_product":       find(["about_product", "about", "description"]),
        "user_id":             find(["user_id", "userid", "reviewer_id", "customer_id"]),
        "user_name":           find(["user_name", "username", "reviewer_name"]),
        "review_title":        find(["review_title", "review", "review_heading"]),
        "img_link":            find(["img_link", "image_link", "img_url"]),
        "product_link":        find(["product_link", "link", "product_url"]),
    }

def get_val(row, col_map, key, default=""):
    col = col_map.get(key)
    if col and col in row:
        return row[col]
    return default


# =============================================================================
# HELPERS
# =============================================================================

def _parse_price(s):
    try:
        return float(re.sub(r"[^\d.]", "", str(s)))
    except Exception:
        return 0.0

def _parse_discount(s):
    try:
        return float(re.sub(r"[^\d.]", "", str(s)))
    except Exception:
        return 0.0

def _parse_rating(s):
    try:
        val = float(str(s).strip().split()[0])
        return min(5.0, max(0.0, val))
    except Exception:
        return 0.0

def _parse_count(s):
    try:
        return int(re.sub(r"[^\d]", "", str(s)))
    except Exception:
        return 0

def _extract_tags(name, cat_parts):
    tags       = set()
    name_lower = str(name).lower()
    for part in cat_parts:
        tags.add(re.sub(r"[^a-z0-9]", "", part.lower()))
    brands = [
        "samsung", "sony", "apple", "lg", "mi", "realme", "oneplus", "hp",
        "dell", "lenovo", "asus", "bosch", "philips", "havells", "prestige",
        "anker", "boat", "jbl", "amazon", "logitech", "ptron", "ambrane",
        "tp-link", "sandisk", "seagate", "wd", "boya", "noise", "zebronics",
    ]
    for brand in brands:
        if brand in name_lower:
            tags.add(brand)
    feature_map = {
        "wireless":      ["wireless", "wifi", "wi-fi", "bluetooth"],
        "fast-charging": ["fast charg", "quick charg", "pd charging", "type-c", "type c"],
        "smart":         ["smart", "alexa", "google assistant", "voice control"],
        "4k":            ["4k", "uhd", "ultra hd"],
        "gaming":        ["gaming", "game", "rgb"],
        "premium":       ["premium", "pro", "ultra", "max", "elite"],
        "portable":      ["portable", "compact", "mini", "lightweight"],
        "waterproof":    ["waterproof", "water resistant", "ip67", "ip68"],
        "led":           ["led", "light", "lamp", "bulb"],
        "kitchen":       ["cooker", "mixer", "grinder", "blender", "microwave", "induction"],
        "audio":         ["headphone", "earphone", "speaker", "earbuds", "headset", "soundbar"],
        "cable":         ["cable", "wire", "charger", "adapter", "usb"],
        "monitor":       ["monitor", "display", "screen"],
        "keyboard":      ["keyboard"],
        "mouse":         ["mouse", "mousepad"],
        "storage":       ["ssd", "hdd", "flash drive", "memory card", "pendrive"],
        "networking":    ["router", "wifi adapter", "ethernet", "modem"],
    }
    for tag, keywords in feature_map.items():
        if any(kw in name_lower for kw in keywords):
            tags.add(tag)
    return list(tags)

def _infer_action(review_title, rating_str):
    positive = ["excellent", "great", "love", "amazing", "perfect", "best",
                "highly recommend", "awesome", "fantastic", "satisfied",
                "good quality", "value for money", "superb", "outstanding", "happy"]
    negative = ["bad", "poor", "worst", "defective", "return", "disappointed",
                "waste", "broken", "useless", "terrible", "horrible", "fake"]
    text   = str(review_title).lower()
    rating = _parse_rating(rating_str)
    if rating >= 4.5 or (rating >= 4.0 and any(w in text for w in positive)):
        return "purchase"
    elif rating >= 3.5:
        return "cart"
    elif any(w in text for w in negative) or rating < 2.5:
        return "view"
    else:
        return "view"

def _name_similarity(a, b):
    def bigrams(s):
        return set(s[i:i+2] for i in range(len(s) - 1))
    ba, bb = bigrams(a), bigrams(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)

def _deduplicate(scored_list):
    seen   = []
    result = []
    for item in scored_list:
        p    = item["product"]
        name = p.get("canonical_name", p["name"].lower()[:60])
        if not any(_name_similarity(name, s) > 0.75 for s in seen):
            result.append(item)
            seen.append(name)
    return result

def _apply_diversity(scored_list, max_per_subcat=2):
    counts = defaultdict(int)
    result = []
    for item in scored_list:
        sub = item["product"].get("subcategory", "Other")
        if counts[sub] < max_per_subcat:
            result.append(item)
            counts[sub] += 1
    return result


# =============================================================================
# DATA LOADER
# =============================================================================

def load_amazon_dataset(filepath):
    products      = {}
    interactions  = []
    seen_products = set()

    with open(filepath, encoding="utf-8", errors="ignore") as f:
        reader  = csv.DictReader(f)
        col_map = build_col_map(reader.fieldnames)
        print(f"  [Columns] {[c for c in col_map.values() if c]}")

        for row in reader:
            pid = get_val(row, col_map, "product_id").strip()
            if not pid:
                pid = get_val(row, col_map, "product_name").strip()[:50]
            if not pid:
                continue

            if pid not in seen_products:
                seen_products.add(pid)

                cat_raw   = get_val(row, col_map, "category")
                cat_parts = [c.strip() for c in cat_raw.split("|") if c.strip()]
                if not cat_parts:
                    cat_parts = [c.strip() for c in cat_raw.split(">") if c.strip()]
                if not cat_parts:
                    cat_parts = [cat_raw.strip()] if cat_raw.strip() else ["Other"]

                top_cat  = cat_parts[0]
                sub_cat  = cat_parts[1] if len(cat_parts) > 1 else top_cat
                leaf_cat = cat_parts[-1]
                name     = get_val(row, col_map, "product_name")
                about    = get_val(row, col_map, "about_product")

                products[pid] = {
                    "name":             name[:120],
                    "canonical_name":   re.sub(r"\s+", " ", name.lower().strip())[:60],
                    "category":         top_cat,
                    "subcategory":      sub_cat,
                    "leaf_category":    leaf_cat,
                    "cat_path":         cat_parts,
                    "discounted_price": _parse_price(get_val(row, col_map, "discounted_price")),
                    "actual_price":     _parse_price(get_val(row, col_map, "actual_price")),
                    "discount_pct":     _parse_discount(get_val(row, col_map, "discount_percentage")),
                    "rating":           _parse_rating(get_val(row, col_map, "rating")),
                    "rating_count":     _parse_count(get_val(row, col_map, "rating_count")),
                    "about":            about[:300] if about else "",
                    "tags":             _extract_tags(name, cat_parts),
                    "img_link":         get_val(row, col_map, "img_link"),
                    "product_link":     get_val(row, col_map, "product_link"),
                }

            uid_raw  = get_val(row, col_map, "user_id")
            name_raw = get_val(row, col_map, "user_name")
            rev_raw  = get_val(row, col_map, "review_title")
            rating_s = get_val(row, col_map, "rating")

            user_ids   = [u.strip() for u in uid_raw.split(",")  if u.strip()] if uid_raw  else []
            user_names = [n.strip() for n in name_raw.split(",") if n.strip()] if name_raw else []
            reviews    = [r.strip() for r in rev_raw.split(",")  if r.strip()] if rev_raw  else []

            for i, uid in enumerate(user_ids):
                if len(uid) < 3:
                    continue
                review_text = reviews[i] if i < len(reviews) else ""
                action = _infer_action(review_text, rating_s)
                uname  = user_names[i] if i < len(user_names) else "User"
                interactions.append({
                    "user_id":    uid,
                    "user_name":  uname,
                    "product_id": pid,
                    "action":     action,
                    "review":     review_text,
                })

    return products, interactions


# =============================================================================
# KNOWLEDGE BASE
# =============================================================================

class KnowledgeBase:
    def __init__(self, csv_path,
                 profile_path="user_profiles.json",
                 sim_path="sim_matrix.json"):
        print("  [Loading] Parsing dataset...")
        self.products, self._raw_interactions = load_amazon_dataset(csv_path)
        self.categories      = self._build_categories()
        self.user_profiles   = {}
        self.interaction_log = []
        self.profile_path    = profile_path
        self.sim_path        = sim_path

        if os.path.exists(profile_path):
            self._load_profiles()
            print(f"  [Loaded] Restored {len(self.user_profiles):,} saved profiles")
        else:
            print(f"  [Loading] Building profiles from {len(self._raw_interactions):,} interactions...")
            for event in self._raw_interactions:
                self.update_user_profile(
                    event["user_id"], event["product_id"], event["action"], silent=True)
            self._save_profiles()

        if os.path.exists(sim_path):
            print(f"  [Loading] Restoring similarity matrix from {sim_path}...")
            self._load_sim_matrix()
        else:
            print("  [Loading] Computing similarity matrix (top-200 active users)...")
            self.sim_matrix = self._build_similarity_matrix(top_n=200)
            self._save_sim_matrix()

        print(f"  [OK] {len(self.products):,} products | {len(self.user_profiles):,} users | "
              f"{len(self._raw_interactions):,} interactions")

    def _build_similarity_matrix(self, top_n=200):
        active   = sorted(self.user_profiles.items(),
                          key=lambda x: len(x[1].get("purchased", [])),
                          reverse=True)[:top_n]
        sim      = defaultdict(list)
        user_ids = [uid for uid, _ in active]
        profiles = {uid: set(p.get("purchased", [])) for uid, p in active}

        for i, uid_a in enumerate(user_ids):
            for uid_b in user_ids[i + 1:]:
                a = profiles[uid_a]
                b = profiles[uid_b]
                if not a or not b:
                    continue
                inter = len(a & b)
                if inter == 0:
                    continue
                jaccard = inter / len(a | b)
                sim[uid_a].append((uid_b, jaccard))
                sim[uid_b].append((uid_a, jaccard))

        for uid in sim:
            sim[uid].sort(key=lambda x: -x[1])
            sim[uid] = sim[uid][:50]
        return sim

    def _save_sim_matrix(self):
        with open(self.sim_path, "w") as f:
            json.dump({uid: neighbours for uid, neighbours in self.sim_matrix.items()}, f)
        print(f"  [Saved] Similarity matrix → {self.sim_path}")

    def _load_sim_matrix(self):
        with open(self.sim_path) as f:
            raw = json.load(f)
        self.sim_matrix = defaultdict(list,
            {uid: [tuple(pair) for pair in neighbours]
             for uid, neighbours in raw.items()})

    def _save_profiles(self):
        serializable = {
            uid: {
                "viewed":          p["viewed"],
                "carted":          p["carted"],
                "purchased":       p["purchased"],
                "category_scores": dict(p["category_scores"]),
                "tag_scores":      dict(p["tag_scores"]),
                "price_history":   p["price_history"],
            }
            for uid, p in self.user_profiles.items()
        }
        with open(self.profile_path, "w") as f:
            json.dump(serializable, f)
        print(f"  [Saved] Profiles → {self.profile_path}")

    def _load_profiles(self):
        with open(self.profile_path) as f:
            data = json.load(f)
        for uid, p in data.items():
            self.user_profiles[uid] = {
                "viewed":          p.get("viewed", []),
                "carted":          p.get("carted", []),
                "purchased":       p.get("purchased", []),
                "category_scores": defaultdict(float, p.get("category_scores", {})),
                "tag_scores":      defaultdict(float, p.get("tag_scores", {})),
                "price_history":   p.get("price_history", []),
            }

    def _build_categories(self):
        cat_data = defaultdict(lambda: {"subcategories": set(), "count": 0, "prices": []})
        for p in self.products.values():
            c = p["category"]
            cat_data[c]["subcategories"].add(p["subcategory"])
            cat_data[c]["count"] += 1
            if p["discounted_price"] > 0:
                cat_data[c]["prices"].append(p["discounted_price"])
        result = {}
        for cat, data in cat_data.items():
            prices = data["prices"]
            result[cat] = {
                "subcategories": list(data["subcategories"]),
                "product_count": data["count"],
                "avg_price":     round(sum(prices) / len(prices), 2) if prices else 0,
            }
        return result

    def update_user_profile(self, user_id, product_id, action, silent=False):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "viewed":          [],
                "carted":          [],
                "purchased":       [],
                "category_scores": defaultdict(float),
                "tag_scores":      defaultdict(float),
                "price_history":   [],
            }
        product = self.products.get(product_id)
        if not product:
            return

        weight   = {"view": 1.0, "cart": 3.0, "purchase": 5.0}.get(action, 1.0)
        profile  = self.user_profiles[user_id]
        list_key = {"view": "viewed", "cart": "carted", "purchase": "purchased"}.get(action, "viewed")

        if product_id not in profile[list_key]:
            profile[list_key].append(product_id)

        profile["category_scores"][product["category"]]      += weight
        profile["category_scores"][product["subcategory"]]   += weight * 0.6
        profile["category_scores"][product["leaf_category"]] += weight * 0.4
        for tag in product.get("tags", []):
            profile["tag_scores"][tag] += weight * 0.35
        if product["discounted_price"] > 0:
            profile["price_history"].append(product["discounted_price"])

        if not silent:
            self.interaction_log.append({
                "user_id":    user_id,
                "product_id": product_id,
                "action":     action,
                "timestamp":  datetime.now().isoformat(),
            })

    def get_product(self, pid):
        return self.products.get(pid)

    def get_all_product_ids(self):
        return list(self.products.keys())


# =============================================================================
# RL STATE ENCODER
# =============================================================================
# The state captures WHAT the user prefers, so the RL agent can learn
# which products to recommend in that context.

class StateEncoder:
    """
    Encodes a user profile into a discrete state string (for tabular Q-learning)
    and a continuous feature vector (for function-approximation Q-learning).

    State dimensions:
      - top_category   : most interacted category (bucketed)
      - price_bucket   : avg spend level  (low / mid / high)
      - engagement     : number of purchases  (new / light / active / heavy)
      - top_tag        : single dominant tag preference
    """

    PRICE_BUCKETS = [500, 2000, 8000]          # ₹ thresholds
    ENGAGEMENT_BUCKETS = [1, 5, 15]            # purchase count thresholds

    def encode_state(self, profile: dict) -> str:
        """Return a hashable discrete state key."""
        # Top category
        cat_scores = profile.get("category_scores", {})
        top_cat = max(cat_scores, key=cat_scores.get) if cat_scores else "none"
        top_cat = re.sub(r"[^a-z0-9]", "", top_cat.lower())[:20]

        # Price bucket
        ph = profile.get("price_history", [])
        avg_price = sum(ph) / len(ph) if ph else 0
        pb = 0
        for thresh in self.PRICE_BUCKETS:
            if avg_price > thresh:
                pb += 1
        price_bucket = ["budget", "low", "mid", "high"][pb]

        # Engagement level
        n_purchases = len(profile.get("purchased", []))
        eb = 0
        for thresh in self.ENGAGEMENT_BUCKETS:
            if n_purchases > thresh:
                eb += 1
        engagement = ["new", "light", "active", "heavy"][eb]

        # Top tag
        tag_scores = profile.get("tag_scores", {})
        top_tag = max(tag_scores, key=tag_scores.get) if tag_scores else "none"
        top_tag = re.sub(r"[^a-z0-9]", "", top_tag.lower())[:15]

        return f"{top_cat}|{price_bucket}|{engagement}|{top_tag}"

    def encode_features(self, product: dict, profile: dict) -> np.ndarray:
        """
        Return a numeric feature vector for linear Q approximation.
        Used when exact state-action pair hasn't been seen before.

        Features (8 dims):
          0  content_score   – category/tag match
          1  price_match     – how close price is to user avg
          2  rating_norm     – product rating / 5
          3  popularity      – log(rating_count) / 6
          4  discount_norm   – discount % / 100
          5  collab_signal   – 1 if similar users bought this
          6  tag_overlap     – fraction of product tags in user tag_scores
          7  brand_loyalty   – 1 if user has bought same brand before
        """
        # Guard: return zero vector if product is missing or empty
        if not product:
            return np.zeros(8, dtype=np.float32)

        cat_s = profile.get("category_scores", {})
        tag_s = profile.get("tag_scores", {})
        ph    = profile.get("price_history", [])

        # 0: content score (normalised to [0,1])
        raw = (cat_s.get(product.get("category", ""), 0) * 0.5 +
               cat_s.get(product.get("subcategory", ""), 0) * 0.35 +
               cat_s.get(product.get("leaf_category", ""), 0) * 0.25 +
               sum(tag_s.get(t, 0) for t in product.get("tags", [])) * 0.5)
        max_cat = max(cat_s.values()) if cat_s else 1.0
        content = min(raw / (max_cat + 1e-9), 1.0)

        # 1: price match
        avg_p  = sum(ph) / len(ph) if ph else 0
        pp     = product.get("discounted_price", 0)
        price_match = (min(avg_p, pp) / max(avg_p, pp, 1)) if pp > 0 else 0.5

        # 2: rating normalised
        rating_norm = product.get("rating", 0) / 5.0

        # 3: popularity
        popularity = math.log10(max(product.get("rating_count", 1), 1) + 1) / 6.0

        # 4: discount
        discount_norm = min(product.get("discount_pct", 0) / 100.0, 1.0)

        # 5: tag overlap
        prod_tags = set(product.get("tags", []))
        user_tags = set(tag_s.keys())
        tag_overlap = len(prod_tags & user_tags) / max(len(prod_tags), 1)

        # 6–7 placeholders (collab & brand will be filled by RL agent if available)
        collab_signal = 0.0
        brand_loyalty = 0.0

        return np.array([content, price_match, rating_norm, popularity,
                         discount_norm, tag_overlap, collab_signal, brand_loyalty],
                        dtype=np.float32)


# =============================================================================
# Q-LEARNING AGENT  (Tabular + Linear-FA fallback)
# =============================================================================

class QLearningAgent:
    """
    Implements tabular Q-Learning with ε-greedy exploration.

    State  : discrete encoding of user profile (StateEncoder.encode_state)
    Action : product_id to recommend
    Reward : 0 (no interaction) | 1 (view) | 3 (cart) | 5 (purchase)

    Q-table: dict  state → {product_id: Q-value}

    For unseen (state, action) pairs we fall back to a dot-product between
    a learned weight vector θ and the feature vector φ(state, action):
       Q(s,a) ≈ θ · φ(s,a)

    Learning update (standard Q-learning / TD(0)):
       Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]

    θ update (gradient descent on TD error):
       θ ← θ + α · δ · φ(s,a)    where δ = TD error
    """

    REWARD_MAP = {"view": 1.0, "cart": 3.0, "purchase": 5.0}
    N_FEATURES = 8

    def __init__(self, alpha=0.1, gamma=0.9,
                 epsilon=0.25, epsilon_min=0.05, epsilon_decay=0.995,
                 q_table_path="q_table.json", theta_path="theta.npy"):

        self.alpha         = alpha          # learning rate
        self.gamma         = gamma          # discount factor
        self.epsilon       = epsilon        # exploration rate
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table_path  = q_table_path
        self.theta_path    = theta_path

        # Q-table: state → {pid: Q-value}
        self.Q: dict[str, dict[str, float]] = defaultdict(dict)

        # Linear FA weight vector (for generalisation to unseen pairs)
        self.theta = np.zeros(self.N_FEATURES, dtype=np.float32)

        self._load()

        self.encoder       = StateEncoder()
        self.update_count  = 0
        self.total_reward  = 0.0
        self.reward_history: list[float] = []

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(self.q_table_path):
            with open(self.q_table_path) as f:
                raw = json.load(f)
            self.Q = defaultdict(dict, {s: dict(a) for s, a in raw.items()})
            print(f"  [RL] Q-table loaded: {len(self.Q):,} states")
        if os.path.exists(self.theta_path):
            self.theta = np.load(self.theta_path)
            print(f"  [RL] θ weights loaded: {self.theta}")

    def save(self):
        with open(self.q_table_path, "w") as f:
            json.dump({s: dict(a) for s, a in self.Q.items()}, f)
        np.save(self.theta_path, self.theta)
        print(f"  [Saved] Q-table ({len(self.Q)} states) + θ weights")

    # ── Q-value lookup ────────────────────────────────────────────────────────

    def _q_value(self, state: str, pid: str, features: np.ndarray) -> float:
        """
        Return Q(state, pid).
        If the pair exists in the table → use that.
        Otherwise fall back to linear function approximation: θ · φ
        """
        if pid in self.Q[state]:
            return self.Q[state][pid]
        return float(np.dot(self.theta, features))

    # ── Action selection (ε-greedy) ────────────────────────────────────────────

    def select_actions(self, state: str, candidates: list[tuple],
                       profile: dict, top_n: int = 5) -> list[str]:
        """
        ε-greedy multi-item selection.

        candidates : list of (pid, product_dict)
        Returns    : list of top_n product IDs (explore or exploit)
        """
        if random.random() < self.epsilon:
            # EXPLORE: pick top_n random unseen products
            sampled = random.sample(candidates, min(top_n * 3, len(candidates)))
            return [pid for pid, _ in sampled[:top_n]]
        else:
            # EXPLOIT: rank by Q-value and return top_n
            scored = []
            for pid, prod in candidates:
                feat = self.encoder.encode_features(prod, profile)
                q    = self._q_value(state, pid, feat)
                scored.append((pid, q))
            scored.sort(key=lambda x: -x[1])
            return [pid for pid, _ in scored[:top_n]]

    # ── Q-learning update ─────────────────────────────────────────────────────

    def update(self, user_id: str, pid: str, action: str,
               profile_before: dict, profile_after: dict,
               candidates_after: list[tuple]) -> float:
        """
        Perform a Q-learning (TD) update.

          state   = encode(profile_before)
          action  = pid
          reward  = REWARD_MAP[action]
          state'  = encode(profile_after)   ← next state after interaction

        Returns the TD error δ.
        """
        state       = self.encoder.encode_state(profile_before)
        next_state  = self.encoder.encode_state(profile_after)
        reward      = self.REWARD_MAP.get(action, 0.0)
        product_obj = self._find_product(pid, candidates_after)
        if not product_obj:
            # Product not in candidates slice — return zero vector, skip update
            return 0.0
        features    = self.encoder.encode_features(product_obj, profile_before)

        # Current Q estimate
        current_q = self._q_value(state, pid, features)

        # Best Q in next state (Bellman target)
        if candidates_after:
            next_qs = [
                self._q_value(
                    next_state, cp,
                    self.encoder.encode_features(cp_prod, profile_after))
                for cp, cp_prod in candidates_after[:100]   # limit for speed
            ]
            best_next_q = max(next_qs) if next_qs else 0.0
        else:
            best_next_q = 0.0

        # TD error
        td_target = reward + self.gamma * best_next_q
        delta     = td_target - current_q

        # ── Tabular update ──
        old_q = self.Q[state].get(pid, float(np.dot(self.theta, features)))
        self.Q[state][pid] = old_q + self.alpha * delta

        # ── Linear-FA update (θ) ──
        self.theta += self.alpha * delta * features
        # Clip θ to prevent blow-up
        self.theta = np.clip(self.theta, -10.0, 10.0)

        # ── Decay ε ──
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.update_count += 1
        self.total_reward  += reward
        self.reward_history.append(reward)

        return delta

    def _find_product(self, pid, candidates):
        for cp, prod in candidates:
            if cp == pid:
                return prod
        # Fallback: look up directly from KB if not in candidates slice
        if hasattr(self, '_kb_ref') and self._kb_ref:
            p = self._kb_ref.get_product(pid)
            if p:
                return p
        return None

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        recent = self.reward_history[-100:] if self.reward_history else [0]
        return {
            "updates":        self.update_count,
            "epsilon":        round(self.epsilon, 4),
            "total_reward":   round(self.total_reward, 1),
            "avg_reward_100": round(sum(recent) / len(recent), 3),
            "q_states":       len(self.Q),
            "theta":          [round(float(v), 4) for v in self.theta],
        }


# =============================================================================
# REINFORCEMENT LEARNING PERFORMANCE ELEMENT
# =============================================================================

class RLPerformanceElement:
    """
    Replaces the old weighted-score PerformanceElement.

    Recommendation pipeline:
      1. Build candidate set (unseen products)
      2. RL agent selects top_n via ε-greedy Q-value ranking
      3. Apply deduplication + diversity post-filters
      4. Return with explanations
    """

    FEATURE_NAMES = [
        "content_match", "price_match", "rating", "popularity",
        "discount", "tag_overlap", "collab", "brand_loyalty"
    ]

    def __init__(self, kb: KnowledgeBase, rl_agent: QLearningAgent):
        self.kb      = kb
        self.agent   = rl_agent
        self.encoder = StateEncoder()

    def _collab_signal(self, pid: str, user_id: str) -> float:
        neighbours = self.kb.sim_matrix.get(user_id, [])
        if not neighbours:
            return 0.0
        w_sum, s_sum = 0.0, 0.0
        for other_id, sim in neighbours[:20]:
            other = self.kb.user_profiles.get(other_id, {})
            if pid in other.get("purchased", []):
                w_sum += sim
            s_sum += sim
        return round(w_sum / s_sum, 4) if s_sum > 0 else 0.0

    def _brand_loyalty(self, pid: str, profile: dict) -> float:
        product     = self.kb.get_product(pid)
        if not product:
            return 0.0
        prod_tags   = set(product.get("tags", []))
        bought_tags = set()
        for bpid in profile.get("purchased", []):
            bp = self.kb.get_product(bpid)
            if bp:
                bought_tags.update(bp.get("tags", []))
        brands = {"samsung", "sony", "apple", "lg", "boat", "anker", "logitech",
                  "hp", "dell", "lenovo", "mi", "realme", "sandisk", "tp-link",
                  "zebronics", "noise"}
        return 1.0 if prod_tags & bought_tags & brands else 0.0

    def _enrich_features(self, pid: str, prod: dict,
                         profile: dict, user_id: str) -> np.ndarray:
        """Build full feature vector including collab + brand."""
        feat = self.encoder.encode_features(prod, profile)
        feat[6] = self._collab_signal(pid, user_id)
        feat[7] = self._brand_loyalty(pid, profile)
        return feat

    def _explain(self, feat: np.ndarray, q_value: float) -> str:
        """Generate a human-readable explanation from the feature vector."""
        idx = int(np.argmax(feat))
        name_map = {
            0: "closely matches your interests",
            1: "fits your usual price range",
            2: "highly rated by customers",
            3: "popular with many shoppers",
            4: "on a great discount",
            5: "matches your tag preferences",
            6: "recommended by similar shoppers",
            7: "from a brand you already trust",
        }
        return name_map.get(idx, "recommended by the RL agent")

    def recommend(self, user_id: str, top_n: int = 5) -> list[dict]:
        """Main recommendation method — fully RL-driven."""
        if user_id not in self.kb.user_profiles:
            return self._cold_start(top_n)

        profile = self.kb.user_profiles[user_id]
        seen    = set(profile.get("viewed", []) +
                      profile.get("carted",  []) +
                      profile.get("purchased", []))

        # Build candidate list
        candidates = [(pid, self.kb.get_product(pid))
                      for pid in self.kb.get_all_product_ids()
                      if pid not in seen and self.kb.get_product(pid)]

        state = self.encoder.encode_state(profile)

        # ε-greedy action selection
        selected_pids = self.agent.select_actions(state, candidates, profile, top_n * 4)

        # Score selected for display + diversity
        scored = []
        for pid in selected_pids:
            prod = self.kb.get_product(pid)
            if not prod:
                continue
            feat    = self._enrich_features(pid, prod, profile, user_id)
            q_val   = self.agent._q_value(state, pid, feat)
            reason  = self._explain(feat, q_val)
            scored.append({
                "product_id":  pid,
                "product":     prod,
                "total_score": round(float(q_val), 4),
                "q_value":     round(float(q_val), 4),
                "features":    {self.FEATURE_NAMES[i]: round(float(feat[i]), 3)
                                for i in range(len(self.FEATURE_NAMES))},
                "epsilon":     round(self.agent.epsilon, 4),
                "mode":        "explore" if self.agent.epsilon > 0.15 else "exploit",
                "reasons":     [reason],
            })

        scored.sort(key=lambda x: -x["total_score"])
        scored = _deduplicate(scored)
        scored = _apply_diversity(scored, max_per_subcat=2)
        return scored[:top_n]

    def _cold_start(self, top_n: int) -> list[dict]:
        """For brand-new users: rank by heuristic (rating × popularity × discount)."""
        scored = []
        for pid in self.kb.get_all_product_ids():
            p = self.kb.get_product(pid)
            if not p:
                continue
            h = (p.get("rating", 0) / 5.0 * 0.5 +
                 math.log10(max(p.get("rating_count", 1), 1) + 1) / 6.0 * 0.35 +
                 min(p.get("discount_pct", 0) / 100.0, 1.0) * 0.15)
            scored.append({
                "product_id":  pid,
                "product":     p,
                "total_score": round(h, 4),
                "q_value":     round(h, 4),
                "features":    {},
                "epsilon":     round(self.agent.epsilon, 4),
                "mode":        "cold-start",
                "reasons":     ["Trending — highly rated by shoppers"],
            })
        scored.sort(key=lambda x: -x["total_score"])
        return _deduplicate(scored)[:top_n]


# =============================================================================
# CRITIC  (unchanged interface, RL-aware logging)
# =============================================================================

class Critic:
    def __init__(self):
        self.log     = []
        self.metrics = defaultdict(lambda: {
            "recommended": 0, "clicked": 0, "carted": 0, "purchased": 0
        })

    def log_recommendation(self, user_id: str, rec_ids: list[str]):
        self.log.append({"user_id": user_id, "recommended": rec_ids})
        self.metrics[user_id]["recommended"] += len(rec_ids)

    def evaluate(self, user_id: str, product_id: str, action: str) -> float:
        recent = {pid for entry in self.log
                  if entry["user_id"] == user_id
                  for pid in entry["recommended"]}
        if product_id not in recent:
            return 0
        m = self.metrics[user_id]
        if action == "view":     m["clicked"]   += 1; return 1.0
        if action == "cart":     m["carted"]    += 1; return 3.0
        if action == "purchase": m["purchased"] += 1; return 5.0
        return 0.0

    def get_report(self, user_id: str) -> dict:
        m     = self.metrics[user_id]
        total = max(m["recommended"], 1)
        return {
            "total_recommendations": m["recommended"],
            "click_through_rate":    round(m["clicked"]   / total, 3),
            "cart_rate":             round(m["carted"]    / total, 3),
            "conversion_rate":       round(m["purchased"] / total, 3),
            "cumulative_reward":     m["clicked"] + m["carted"] * 3 + m["purchased"] * 5,
        }


# =============================================================================
# LEARNING ELEMENT  — now drives Q-learning updates
# =============================================================================

class LearningElement:
    """
    Replaces static weight tuning with Q-learning TD updates.

    After every user interaction:
      1. Record reward from Critic
      2. Compute TD error and update Q(state, action) and θ
      3. Decay ε (exploration)

    Exposes `weights` property for display compatibility (derived from θ).
    """

    def __init__(self, rl_agent: QLearningAgent):
        self.agent   = rl_agent
        self.history: list[dict] = []

    @property
    def weights(self) -> dict:
        """
        Map θ components to named weights for display.
        These are the actual learned feature importances.
        """
        th = self.agent.theta
        names = ["content", "price", "rating", "popularity",
                 "discount", "tag", "collab", "brand"]
        total = sum(abs(v) for v in th) or 1.0
        return {n: round(abs(float(th[i])) / total, 4)
                for i, n in enumerate(names)}

    def update_from_interaction(self, user_id: str, pid: str, action: str,
                                profile_before: dict, profile_after: dict,
                                all_candidates: list[tuple]) -> float:
        """Trigger a Q-learning update and log the result."""
        delta = self.agent.update(
            user_id, pid, action,
            profile_before, profile_after, all_candidates)

        stats = self.agent.stats()
        self.history.append({
            "action":   action,
            "delta":    round(delta, 4),
            "epsilon":  stats["epsilon"],
            "theta":    stats["theta"],
        })
        return delta

    def print_history(self):
        if not self.history:
            print("  No RL updates yet.")
            return
        print(f"\n  RL Update History (last 5 of {len(self.history)}):")
        for h in self.history[-5:]:
            print(f"    action={h['action']:<10}  δ={h['delta']:+.4f}  "
                  f"ε={h['epsilon']:.4f}  θ={h['theta']}")


# =============================================================================
# PROBLEM GENERATOR  (exploration suggestions — unchanged)
# =============================================================================

class ProblemGenerator:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def suggest_explorations(self, user_id: str, n: int = 2) -> list[dict]:
        if user_id not in self.kb.user_profiles:
            return []
        profile  = self.kb.user_profiles[user_id]
        explored = {
            self.kb.get_product(pid)["category"]
            for pid in profile.get("viewed", []) + profile.get("purchased", [])
            if self.kb.get_product(pid)
        }
        unexplored  = set(self.kb.categories.keys()) - explored
        suggestions = []
        for cat in list(unexplored)[:n]:
            candidates = [(pid, p) for pid, p in self.kb.products.items()
                          if p["category"] == cat and p["rating"] >= 4.0]
            if candidates:
                best = max(candidates,
                           key=lambda x: x[1]["rating"] +
                           math.log10(max(x[1]["rating_count"], 1) + 1))
                suggestions.append({
                    "product_id": best[0],
                    "product":    best[1],
                    "reason":     f"Explore {cat.replace('&', ' & ')} — you might love it!",
                })
        return suggestions


# =============================================================================
# MAIN AGENT  — wires everything together
# =============================================================================

class ShoppingRecommendationAgent:
    """
    RL-powered Shopping Recommendation Agent.

    Architecture (Intelligent Agent framework):
      ┌──────────────────────────────────────────────────────────────┐
      │  Environment : Amazon product catalogue + user interactions  │
      │  State       : Encoded user profile (StateEncoder)           │
      │  Action      : Which product to recommend                    │
      │  Reward      : view=+1, cart=+3, purchase=+5                 │
      │  Algorithm   : Q-Learning with ε-greedy + linear FA fallback │
      └──────────────────────────────────────────────────────────────┘

    Components:
      KnowledgeBase       — product data, user profiles, similarity matrix
      QLearningAgent      — Q-table + θ weights + TD update rule
      RLPerformanceElement— translates Q-values into ranked recommendations
      Critic              — measures CTR / conversion / cumulative reward
      LearningElement     — orchestrates Q-learning updates after each event
      ProblemGenerator    — surfaces unexplored categories (exploration bonus)
    """

    def __init__(self, csv_path: str):
        print("\n" + "=" * 68)
        print("  SHOPPING RECOMMENDATION RL AGENT  v3.0")
        print("  Algorithm: Q-Learning  (ε-greedy + Linear FA)")
        print("=" * 68)
        self.kb          = KnowledgeBase(csv_path)
        self.rl_agent    = QLearningAgent()
        self.performance = RLPerformanceElement(self.kb, self.rl_agent)
        self.critic      = Critic()
        self.learner     = LearningElement(self.rl_agent)
        self.explorer    = ProblemGenerator(self.kb)
        print("  [OK] RL Agent ready.\n")

    # ── Core interaction loop ─────────────────────────────────────────────────

    def process_interaction(self, user_id: str, product_id: str, action: str):
        """
        Full RL loop for one (user, product, action) event:
          1. Snapshot profile BEFORE update  →  state s
          2. Update KB profile               →  state s'
          3. Critic evaluates reward
          4. LearningElement performs Q(s,a) TD update
          5. Print feedback
        """
        # 1. Snapshot current state
        profile_before = self._snapshot_profile(user_id)

        # 2. Update knowledge base (next state)
        self.kb.update_user_profile(user_id, product_id, action)
        profile_after = self._snapshot_profile(user_id)

        # 3. Reward
        reward = self.critic.evaluate(user_id, product_id, action)

        # 4. Q-learning update
        all_pids   = self.kb.get_all_product_ids()
        candidates = [(pid, self.kb.get_product(pid))
                      for pid in all_pids[:500]    # cap for speed
                      if self.kb.get_product(pid)]
        # Always ensure the interacted product is in the candidate list
        interacted_prod = self.kb.get_product(product_id)
        if interacted_prod and not any(pid == product_id for pid, _ in candidates):
            candidates.append((product_id, interacted_prod))

        delta = self.learner.update_from_interaction(
            user_id, product_id, action,
            profile_before, profile_after, candidates)

        product = self.kb.get_product(product_id)
        name    = product["name"][:55] if product else "Unknown"
        print(f"  [{action.upper():8}] {user_id[:20]} → {name}  "
              f"(+{reward:.0f} reward | δ={delta:+.4f} | ε={self.rl_agent.epsilon:.4f})")

    def _snapshot_profile(self, user_id: str) -> dict:
        """Return a lightweight copy of the current user profile."""
        p = self.kb.user_profiles.get(user_id, {})
        return {
            "viewed":          list(p.get("viewed",    [])),
            "carted":          list(p.get("carted",    [])),
            "purchased":       list(p.get("purchased", [])),
            "category_scores": dict(p.get("category_scores", {})),
            "tag_scores":      dict(p.get("tag_scores", {})),
            "price_history":   list(p.get("price_history", [])),
        }

    # ── Recommendations ───────────────────────────────────────────────────────

    def get_recommendations(self, user_id: str, top_n: int = 5):
        recs    = self.performance.recommend(user_id, top_n)
        rec_ids = [r["product_id"] for r in recs]
        self.critic.log_recommendation(user_id, rec_ids)
        explorations = self.explorer.suggest_explorations(user_id, n=2)
        return recs, explorations

    def display_recommendations(self, user_id: str, top_n: int = 5):
        print("\n" + "=" * 68)
        print(f"  RL RECOMMENDATIONS FOR: {user_id}")
        print("=" * 68)
        recs, explorations = self.get_recommendations(user_id, top_n)

        profile = self.kb.user_profiles.get(user_id, {})
        p_count = len(profile.get("purchased", []))
        v_count = len(profile.get("viewed",    []))
        c_count = len(profile.get("carted",    []))
        state   = self.performance.encoder.encode_state(profile)
        if profile.get("price_history"):
            avg = sum(profile["price_history"]) / len(profile["price_history"])
            print(f"  Profile : {p_count} purchases | {c_count} carted | "
                  f"{v_count} views | Avg spend: ₹{avg:,.0f}")
        print(f"  RL State: {state}")
        print(f"  ε={self.rl_agent.epsilon:.4f} | "
              f"Q-states={len(self.rl_agent.Q):,} | "
              f"Updates={self.rl_agent.update_count:,}")
        print()

        mode_label = recs[0]["mode"] if recs else "exploit"
        print(f"  TOP {top_n} RECOMMENDATIONS  [{mode_label.upper()}]:")
        print("  " + "-" * 64)
        for i, rec in enumerate(recs, 1):
            p      = rec["product"]
            price  = f"₹{p['discounted_price']:,.0f}" if p["discounted_price"] else "N/A"
            orig   = f"₹{p['actual_price']:,.0f}"     if p["actual_price"]     else ""
            disc   = f" ({p['discount_pct']:.0f}% off)" if p["discount_pct"]  else ""
            reason = rec["reasons"][0] if rec["reasons"] else "Recommended by RL agent"
            feat   = rec.get("features", {})
            print(f"\n  #{i}  {p['name'][:72]}")
            print(f"       Category : {p['category']} > {p['leaf_category']}")
            print(f"       Price    : {price}{disc}  (was {orig})  |  "
                  f"Rating: {p['rating']}/5 ({p['rating_count']:,} reviews)")
            print(f"       Q-value  : {rec['q_value']:.4f}")
            if feat:
                top_feats = sorted(feat.items(), key=lambda x: -x[1])[:3]
                feat_str  = " | ".join(f"{k}={v:.3f}" for k, v in top_feats)
                print(f"       Features : {feat_str}")
            print(f"       Why?     : {reason}")

        if explorations:
            print(f"\n  DISCOVER SOMETHING NEW (Exploration Bonus):")
            print("  " + "-" * 64)
            for exp in explorations:
                p = exp["product"]
                print(f"\n  [EXPLORE]  {p['name'][:72]}")
                print(f"       {p['category']}  |  ₹{p['discounted_price']:,.0f}"
                      f"  |  {p['rating']}/5")
                print(f"       {exp['reason']}")

        print(f"\n  Learned θ weights:")
        for k, v in self.learner.weights.items():
            bar = "█" * int(v * 30)
            print(f"    {k:<14}: {v:.4f}  {bar}")
        print()

    # ── Performance display ───────────────────────────────────────────────────

    def display_performance(self, user_id: str):
        report = self.critic.get_report(user_id)
        rl     = self.rl_agent.stats()
        print(f"\n  PERFORMANCE — {user_id}")
        for k, v in report.items():
            print(f"    {k.replace('_', ' ').title():<28}: {v}")
        print(f"\n  RL AGENT STATS")
        for k, v in rl.items():
            if k != "theta":
                print(f"    {k.replace('_', ' ').title():<28}: {v}")
        return report

    # ── Utilities ─────────────────────────────────────────────────────────────

    def search_products(self, query: str, top_n: int = 10):
        q       = query.lower()
        results = [(pid, p) for pid, p in self.kb.products.items()
                   if q in p["name"].lower() or q in p.get("about", "").lower()]
        results.sort(key=lambda x: x[1]["rating_count"], reverse=True)
        return results[:top_n]

    def list_users(self, top_n: int = 20, min_purchases: int = 1):
        active = [(uid, len(p.get("purchased", [])), len(p.get("viewed", [])))
                  for uid, p in self.kb.user_profiles.items()
                  if len(p.get("purchased", [])) >= min_purchases]
        active.sort(key=lambda x: -x[1])
        return active[:top_n]


# =============================================================================
# INTERACTIVE CLI
# =============================================================================

def run_interactive(agent: ShoppingRecommendationAgent):
    print("\n" + "=" * 68)
    print("  INTERACTIVE MODE  (RL Agent)")
    print("=" * 68)
    print("  recommend <user_id> [n]                   get top-n recommendations")
    print("  interact  <user_id> <product_id> <action> log interaction + RL update")
    print("  search    <keyword>                        find product IDs")
    print("  users     [n]                              list active users")
    print("  rl                                         show RL agent stats & θ")
    print("  perf      <user_id>                        performance metrics")
    print("  save                                       save Q-table & θ")
    print("  quit                                       exit")
    print()
    print("  Actions: view | cart | purchase")
    print()

    while True:
        try:
            raw = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd   = parts[0].lower()

        if cmd in ("quit", "exit", "q"):
            agent.rl_agent.save()
            agent.kb._save_profiles()
            print("  Goodbye!")
            break

        elif cmd == "recommend" and len(parts) >= 2:
            uid = parts[1]
            n   = int(parts[2]) if len(parts) >= 3 else 5
            agent.display_recommendations(uid, top_n=n)

        elif cmd == "interact" and len(parts) == 4:
            uid, pid, action = parts[1], parts[2], parts[3].lower()
            if action not in ("view", "cart", "purchase"):
                print("  Action must be: view, cart, or purchase")
            else:
                agent.process_interaction(uid, pid, action)
                agent.kb._save_profiles()

        elif cmd == "search" and len(parts) >= 2:
            query   = " ".join(parts[1:])
            results = agent.search_products(query)
            if not results:
                print(f"  No products found for '{query}'")
            else:
                print(f"\n  Results for '{query}':")
                for pid, p in results:
                    print(f"    {pid}  |  {p['name'][:55]}  |  "
                          f"₹{p['discounted_price']:,.0f}  |  {p['rating']}/5")

        elif cmd == "users":
            n     = int(parts[1]) if len(parts) >= 2 else 15
            users = agent.list_users(top_n=n)
            print(f"\n  Top {n} active users:")
            for uid, purchases, views in users:
                print(f"    {uid}  |  {purchases} purchases  |  {views} views")

        elif cmd == "rl":
            stats = agent.rl_agent.stats()
            print(f"\n  RL AGENT STATS:")
            for k, v in stats.items():
                if k == "theta":
                    names = ["content", "price", "rating", "pop",
                             "discount", "tag", "collab", "brand"]
                    print(f"    {'theta':<28}: " +
                          " ".join(f"{n}={v:.3f}" for n, v in zip(names, v)))
                else:
                    print(f"    {k.replace('_',' ').title():<28}: {v}")
            agent.learner.print_history()

        elif cmd == "perf" and len(parts) >= 2:
            agent.display_performance(parts[1])

        elif cmd == "save":
            agent.rl_agent.save()
            agent.kb._save_profiles()
            print("  Saved.")

        else:
            print("  Unknown command.")
            print("  Try: recommend | interact | search | users | rl | perf | save | quit")


# =============================================================================
# DEMO
# =============================================================================

def run_demo(csv_path: str):
    agent = ShoppingRecommendationAgent(csv_path)

    print("  [Demo] Simulating live interactions for RL warm-up...")
    top_buyers = sorted(
        [(uid, p) for uid, p in agent.kb.user_profiles.items()
         if len(p.get("purchased", [])) >= 3],
        key=lambda x: -len(x[1].get("purchased", []))
    )

    count = 0
    for uid, profile in top_buyers[:15]:          # more users for better Q coverage
        for pid in profile.get("purchased", [])[:5]:
            agent.process_interaction(uid, pid, "purchase")
            count += 1
        for pid in profile.get("carted", [])[:3]:
            agent.process_interaction(uid, pid, "cart")
            count += 1
        for pid in profile.get("viewed", [])[:3]:
            agent.process_interaction(uid, pid, "view")
            count += 1

    print(f"  [Demo] {count} interactions → {agent.rl_agent.update_count} Q-updates.\n")

    demo_users = [uid for uid, _ in top_buyers[:3]]
    if not demo_users:
        demo_users = list(agent.kb.user_profiles.keys())[:3]

    print("\n" + "=" * 68)
    print("  DEMO: RL Recommendations for Real Users")
    print("=" * 68)
    for uid in demo_users:
        agent.display_recommendations(uid, top_n=5)

    print("\n" + "=" * 68)
    print("  DEMO: New User (cold start)")
    print("=" * 68)
    agent.display_recommendations("NEW_USER_XYZ", top_n=5)

    print("\n" + "=" * 68)
    print("  PERFORMANCE REPORTS")
    print("=" * 68)
    for uid in demo_users[:3]:
        agent.display_performance(uid)

    agent.learner.print_history()
    agent.rl_agent.save()

    print("\n" + "=" * 68)
    print("  DATASET & RL STATISTICS")
    print("=" * 68)
    cat_counts = Counter(p["category"] for p in agent.kb.products.values())
    rl_stats   = agent.rl_agent.stats()
    print(f"  Products       : {len(agent.kb.products):,}")
    print(f"  Users          : {len(agent.kb.user_profiles):,}")
    print(f"  Events         : {len(agent.kb._raw_interactions):,}")
    print(f"  Q-states       : {rl_stats['q_states']:,}")
    print(f"  Q-updates      : {rl_stats['updates']:,}")
    print(f"  ε (exploration): {rl_stats['epsilon']:.4f}")
    print(f"  Avg reward/100 : {rl_stats['avg_reward_100']:.3f}")
    print(f"  Categories:")
    for cat, cnt in cat_counts.most_common():
        print(f"    {cat:<35}: {cnt:,} products")
    print("\n[DEMO COMPLETE]\n")
    return agent


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    args       = sys.argv[1:]
    quick_mode = "--interact" in args
    path_args  = [a for a in args if not a.startswith("--")]

    if path_args:
        path = path_args[0]
    else:
        candidates = ["amazon.csv", "amazon_data/amazon.csv"]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if not path:
            print("ERROR: amazon.csv not found.")
            print("Usage: python agent_rl.py <amazon.csv> [--interact]")
            sys.exit(1)

    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    if quick_mode:
        print("\n" + "=" * 68)
        print("  RL SHOPPING AGENT  v3.0  —  Quick Interactive Mode")
        print("=" * 68)
        agent = ShoppingRecommendationAgent(path)
        run_interactive(agent)
    else:
        agent = run_demo(path)
        run_interactive(agent)
