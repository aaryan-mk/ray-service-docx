# Profile Search Integration: Direct Milvus + MongoDB + S3

**Last Updated:** February 22, 2026

---

## Purpose

After face-ocr-api processes a video and clusters faces into identities, we want to automatically search for matching social media profiles from a database of **1M+ profiles** (Twitter, Instagram, TikTok).

This document describes the **direct bypass** approach — connecting to Milvus, MongoDB, and S3 directly from our Python service, instead of going through the broken media-recognizer Go service.

---

## Why Bypass media-recognizer?

The media-recognizer Go service wraps Milvus + MongoDB behind an HTTP API. It's currently broken:

| Approach | Status | Latency |
|----------|--------|---------|
| media-recognizer Go API | Broken (60s timeout, 500 error) | N/A |
| **Direct pymilvus + pymongo** | **Working** | **~2s** |

**Root cause:** The `buffalo_l` Milvus collection has 1,024 partitions. The Go Milvus SDK's partition discovery mechanism can't handle this — it times out after 60s trying to query each partition. The Python SDK (pymilvus) handles it transparently.

**Proven on Feb 22, 2026:** Full end-to-end test from WARP container — Milvus search (0.4s) + MongoDB metadata (1.7s) + S3 image download (2.2s) = 5.3s total, all 5 matches returned with full profile data.

---

## Architecture

### Before (Broken)

```
face-ocr-api  →  media-recognizer (Go)  →  Milvus + MongoDB
                        ❌ 60s timeout
```

### After (Direct Access)

```
face-ocr-api
    │
    ├── pymilvus  ────→  Milvus (172.31.30.225:19530)
    │                     buffalo_l collection
    │                     1,021,778 faces, 512-D ArcFace, COSINE
    │                     Search: ~400ms
    │
    ├── pymongo   ────→  MongoDB (172.31.30.225:27017)
    │                     media_recognizer.GenericFeatures
    │                     1,263,495 profile docs
    │                     Lookup: ~1.7s (batch)
    │
    └── boto3     ────→  S3 (s3.us-east-1.amazonaws.com)
                          milvus-storage-2025 bucket
                          Profile photos (20-40KB each)
                          Download: ~2s (batch of 5)
```

**Network:** All 3 services are behind the AWS VPN. The face-ocr-api pod needs a **WARP VPN sidecar** (same as media-recognizer-aws-milvus already uses) to reach `172.31.30.225`.

---

## What's in the Databases

### Milvus — `buffalo_l` Collection

| Field | Type | Description |
|-------|------|-------------|
| `_id` | INT64 | Primary key (auto) |
| `_entity_id` | VARCHAR(255) | Links to MongoDB (Trie index) |
| `_partition_id` | VARCHAR(255) | Partition routing (Trie index) |
| `features` | FLOAT_VECTOR(512) | ArcFace embedding (IVF_FLAT, COSINE) |
| `center` | BOOL | True = representative face for this person |

- **1,021,778 entities** across 1,024 partitions
- **Index:** IVF_FLAT with nlist=1024, all rows indexed
- **Metric:** COSINE similarity
- **Mmap enabled** — loads instantly, stable 385ms avg latency

### MongoDB — `media_recognizer.GenericFeatures` Collection

| Field | Type | Description |
|-------|------|-------------|
| `_entity_id` | String | Matches Milvus `_entity_id` (B-tree + text indexed) |
| `_recognizer_id` | String | Which model ingested this (e.g. `siglip2_base_patch16_512._default_0`) |
| `_parent_id` | String/null | Unused (always null) |
| `_group_id` | String/null | Cross-platform group (only 5 entries use this) |
| `metadata` | Object | Platform-specific profile data (see below) |
| `created_at` | DateTime | |
| `updated_at` | DateTime | |

**1,263,495 documents total.**

### MongoDB — Metadata Structure

Each document has exactly one platform key inside `metadata`:

```json
{
  "_entity_id": "089cac6fdc9b1267246b74d67a774f36f8420f8b",
  "metadata": {
    "twitterData": {
      "name": "Pericles 'Perry' Abbasi",
      "id": "2359971372",
      "profileUrl": "https://twitter.com/ElectionLegal",
      "profileImage": "profile_images/089cac6f...jpg"
    }
  }
}
```

Platform keys: `twitterData`, `tiktokData`, `instagramData`

| Platform | Count | Fields Always Present | Bonus Fields (some entries) |
|----------|-------|-----------------------|-----------------------------|
| **Twitter** | 538,181 | name, id, profileImage, profileUrl | twitterName (53%), username (35%) |
| **Instagram** | 488,821 | name, id, profileImage | profileUrl (70%), biography, follower_count, following_count, media_count, category (18%) |
| **TikTok** | 236,491 | name, id, profileImage | — |

### S3 — Profile Images

| Detail | Value |
|--------|-------|
| **Bucket** | `milvus-storage-2025` |
| **Region** | `us-east-1` |
| **Access Key** | `<AWS_ACCESS_KEY>` (see env vars) |
| **Secret Key** | `<AWS_SECRET_KEY>` (see env vars) |
| **Image sizes** | 15-45KB (JPEG/WebP) |
| **Path pattern** | `profile_images/{entity_id}.jpg` or `.webp` |
| **Presigned URLs** | Blocked by bucket policy — must use SDK direct download |

---

## Embedding Compatibility — Confirmed

Our face-ocr-worker and the `buffalo_l` collection both use ArcFace w600k_r50:

| Property | Our Pipeline | buffalo_l Collection |
|----------|-------------|---------------------|
| Model | ArcFace w600k_r50 | ArcFace w600k_r50 (via InsightFace buffalo_l) |
| Dimensions | 512 | 512 |
| Normalization | L2 normalized (norm = 1.0) | L2 normalized |
| Metric | COSINE | COSINE |

**Validated:** Our embeddings searched against buffalo_l return meaningful results (top match 76.9% vs random noise at 19.9%).

### Similarity Thresholds

| Threshold | Meaning |
|-----------|---------|
| ~20% | Random noise (no match) |
| 60-70% | Weak match (possibly same person, low confidence) |
| **70%+** | **Strong match (recommended threshold)** |
| 80%+ | Very strong match |
| 90%+ | Near-certain match (same photo) |

**Recommendation:** Use **0.70 (70%)** as the minimum similarity threshold. This gives clear separation between random noise (~20%) and true matches (~77%).

---

## Implementation Plan

### Files to Create/Modify

| # | File | Change |
|---|------|--------|
| 1 | `core/config.py` | Add profile search settings (Milvus, MongoDB, S3) |
| 2 | `app/services/profile_search_service.py` | **New** — pymilvus + pymongo + boto3 client |
| 3 | `app/services/state.py` | Add `profile_search_service` reference |
| 4 | `core/server.py` | Init as 6th service in lifespan |
| 5 | `app/database/connection.py` | New `profile_matches` table |
| 6 | `app/database/operations.py` | `save_profile_match()` and `get_profile_matches()` |
| 7 | `app/api/v1/v3_face.py` | Call service in `_store_and_match_identities()` |
| 8 | `app/api/v1/v3_face.py` | New endpoint: `GET /v3/identity/{job_id}/{cluster_id}/profiles` |
| 9 | `requirements.txt` | Add pymilvus, pymongo, boto3 |
| 10 | `draft-deployment/dev/face-ocr-api.yaml` | New env vars + WARP sidecar |

### New Dependencies

Add to `face-ocr-api/requirements.txt`:

```
pymilvus>=2.4.0,<3.0.0
pymongo>=4.6.0
boto3>=1.34.0
```

---

### Step 1: Configuration (`core/config.py`)

Add after the DOSASHOP section (~line 108):

```python
# Profile Search (Direct Milvus + MongoDB + S3)
PROFILE_SEARCH_ENABLED: bool = Field(
    default=False,
    description="Enable profile search via direct Milvus/MongoDB access"
)
PROFILE_SEARCH_MILVUS_URI: str = Field(
    default="http://172.31.30.225:19530",
    description="Milvus server URI for profile vector search"
)
PROFILE_SEARCH_MILVUS_COLLECTION: str = Field(
    default="buffalo_l",
    description="Milvus collection name"
)
PROFILE_SEARCH_MONGO_URI: str = Field(
    default="mongodb://serveradmin:<MONGO_PASSWORD>@<MILVUS_HOST>:27017",
    description="MongoDB connection string for profile metadata"
)
PROFILE_SEARCH_MONGO_DB: str = Field(
    default="media_recognizer",
    description="MongoDB database name"
)
PROFILE_SEARCH_MONGO_COLLECTION: str = Field(
    default="GenericFeatures",
    description="MongoDB collection for profile metadata"
)
PROFILE_SEARCH_S3_BUCKET: str = Field(
    default="milvus-storage-2025",
    description="S3 bucket for profile images"
)
PROFILE_SEARCH_S3_REGION: str = Field(
    default="us-east-1",
    description="S3 region"
)
PROFILE_SEARCH_S3_ACCESS_KEY: str = Field(
    default="",
    description="S3 access key for profile images"
)
PROFILE_SEARCH_S3_SECRET_KEY: str = Field(
    default="",
    description="S3 secret key for profile images"
)
PROFILE_SEARCH_MIN_SIMILARITY: float = Field(
    default=0.70,
    description="Minimum cosine similarity for profile matching (0-1)"
)
PROFILE_SEARCH_TOP_K: int = Field(
    default=5,
    description="Number of top profile matches to return per identity"
)
PROFILE_SEARCH_NPROBE: int = Field(
    default=16,
    description="Milvus nprobe parameter (higher = more accurate, slower)"
)
```

---

### Step 2: Service (`app/services/profile_search_service.py`)

**New file.** Follows the same pattern as `chroma_service.py` — class with `connect()`, business methods, `close()`.

```python
"""
Profile Search Service — Direct Milvus + MongoDB + S3.

Searches for matching social media profiles by querying:
1. Milvus (buffalo_l) — vector similarity search on ArcFace embeddings
2. MongoDB (GenericFeatures) — profile metadata (name, platform, URLs)
3. S3 (milvus-storage-2025) — profile photos

Bypasses the broken media-recognizer Go service.
"""

import logging
import asyncio
import base64
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from core.config import settings

logger = logging.getLogger(__name__)

# Platform keys in MongoDB metadata → normalized platform names
PLATFORM_MAP = {
    'twitterData': 'twitter',
    'tiktokData': 'tiktok',
    'instagramData': 'instagram',
    'facebookData': 'facebook',
}


class ProfileSearchService:
    """Direct Milvus + MongoDB + S3 profile search."""

    def __init__(self):
        self.milvus_client = None
        self.mongo_client = None
        self.mongo_collection = None
        self.s3_client = None
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def connect(self) -> bool:
        """Initialize all 3 clients and verify connectivity."""
        loop = asyncio.get_running_loop()
        try:
            # All 3 clients are sync — initialize in executor
            ok = await loop.run_in_executor(self._executor, self._connect_sync)
            return ok
        except Exception as e:
            logger.error(f"Profile search connection failed: {e}")
            return False

    def _connect_sync(self) -> bool:
        """Synchronous connection setup (runs in thread pool)."""
        from pymilvus import MilvusClient
        from pymongo import MongoClient
        import boto3

        # 1. Milvus
        self.milvus_client = MilvusClient(uri=settings.PROFILE_SEARCH_MILVUS_URI)
        collection = settings.PROFILE_SEARCH_MILVUS_COLLECTION

        # Verify collection exists and is loaded
        collections = self.milvus_client.list_collections()
        if collection not in collections:
            logger.error(f"Milvus collection '{collection}' not found. Available: {collections}")
            return False

        stats = self.milvus_client.get_collection_stats(collection)
        row_count = stats.get('row_count', 0)
        logger.info(f"Milvus connected: {collection} ({row_count:,} vectors)")

        # 2. MongoDB
        self.mongo_client = MongoClient(
            settings.PROFILE_SEARCH_MONGO_URI,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=30000,
        )
        # Verify connectivity
        self.mongo_client.admin.command('ping')
        db = self.mongo_client[settings.PROFILE_SEARCH_MONGO_DB]
        self.mongo_collection = db[settings.PROFILE_SEARCH_MONGO_COLLECTION]
        doc_count = self.mongo_collection.estimated_document_count()
        logger.info(f"MongoDB connected: {settings.PROFILE_SEARCH_MONGO_DB}.{settings.PROFILE_SEARCH_MONGO_COLLECTION} ({doc_count:,} docs)")

        # 3. S3
        if settings.PROFILE_SEARCH_S3_ACCESS_KEY:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.PROFILE_SEARCH_S3_ACCESS_KEY,
                aws_secret_access_key=settings.PROFILE_SEARCH_S3_SECRET_KEY,
                region_name=settings.PROFILE_SEARCH_S3_REGION,
            )
            # Verify bucket access
            self.s3_client.head_bucket(Bucket=settings.PROFILE_SEARCH_S3_BUCKET)
            logger.info(f"S3 connected: {settings.PROFILE_SEARCH_S3_BUCKET}")
        else:
            logger.warning("S3 credentials not configured — profile images will not be available")

        return True

    async def search_profiles(
        self,
        embedding: List[float],
        top_k: int = None,
        min_similarity: float = None,
        include_image: bool = False,
    ) -> List[Dict]:
        """
        Search for matching social media profiles.

        Args:
            embedding: 512-D L2-normalized ArcFace face embedding
            top_k: Max results (default: from settings)
            min_similarity: Min cosine similarity 0-1 (default: from settings)
            include_image: If True, download profile images as base64

        Returns:
            List of profile matches, each with:
            - entity_id, similarity, platform, name, profile_id,
              profile_url, profile_image_path, profile_image_base64 (if requested)
        """
        loop = asyncio.get_running_loop()
        top_k = top_k or settings.PROFILE_SEARCH_TOP_K
        min_similarity = min_similarity or settings.PROFILE_SEARCH_MIN_SIMILARITY

        try:
            # Step 1: Milvus vector search (sync, in executor)
            milvus_results = await loop.run_in_executor(
                self._executor,
                lambda: self._search_milvus(embedding, top_k)
            )

            if not milvus_results:
                return []

            # Filter by similarity threshold
            milvus_results = [
                r for r in milvus_results
                if r['similarity'] >= min_similarity
            ]

            if not milvus_results:
                return []

            # Step 2: MongoDB metadata lookup (sync, in executor)
            entity_ids = [r['entity_id'] for r in milvus_results]
            metadata_map = await loop.run_in_executor(
                self._executor,
                lambda: self._lookup_metadata(entity_ids)
            )

            # Step 3: Combine results
            matches = []
            for result in milvus_results:
                entity_id = result['entity_id']
                meta = metadata_map.get(entity_id, {})

                # Extract platform-specific profile data
                for platform_key, platform_name in PLATFORM_MAP.items():
                    platform_data = meta.get(platform_key)
                    if not platform_data:
                        continue

                    match = {
                        'entity_id': entity_id,
                        'similarity': result['similarity'],
                        'platform': platform_name,
                        'name': platform_data.get('name', ''),
                        'profile_id': platform_data.get('id', ''),
                        'profile_url': platform_data.get('profileUrl', ''),
                        'profile_image_path': platform_data.get('profileImage', ''),
                    }

                    # Optional: extra Instagram fields
                    if platform_name == 'instagram':
                        for extra_field in ['biography', 'follower_count', 'following_count',
                                           'media_count', 'category', 'full_name', 'external_url']:
                            if extra_field in platform_data:
                                match[extra_field] = platform_data[extra_field]

                    matches.append(match)

            # Step 4 (optional): Download profile images from S3
            if include_image and self.s3_client and matches:
                image_map = await loop.run_in_executor(
                    self._executor,
                    lambda: self._download_images([m['profile_image_path'] for m in matches])
                )
                for match in matches:
                    img_path = match['profile_image_path']
                    if img_path in image_map:
                        img_bytes = image_map[img_path]
                        ext = 'jpeg' if img_path.endswith('.jpg') else 'webp'
                        match['profile_image_base64'] = (
                            f"data:image/{ext};base64,"
                            + base64.b64encode(img_bytes).decode()
                        )

            return matches

        except Exception as e:
            logger.error(f"Profile search failed: {e}")
            return []

    def _search_milvus(self, embedding: List[float], top_k: int) -> List[Dict]:
        """Synchronous Milvus vector search."""
        results = self.milvus_client.search(
            collection_name=settings.PROFILE_SEARCH_MILVUS_COLLECTION,
            data=[embedding],
            limit=top_k,
            output_fields=['_entity_id'],
            search_params={
                'metric_type': 'COSINE',
                'params': {'nprobe': settings.PROFILE_SEARCH_NPROBE}
            },
        )

        matches = []
        for hit in results[0]:
            matches.append({
                'entity_id': hit['entity']['_entity_id'],
                'similarity': round(hit['distance'], 4),
            })
        return matches

    def _lookup_metadata(self, entity_ids: List[str]) -> Dict[str, Dict]:
        """Synchronous MongoDB batch metadata lookup."""
        docs = self.mongo_collection.find(
            {'_entity_id': {'$in': entity_ids}},
            {'_entity_id': 1, 'metadata': 1, '_id': 0}
        )
        return {
            doc['_entity_id']: doc.get('metadata', {})
            for doc in docs
        }

    def _download_images(self, image_paths: List[str]) -> Dict[str, bytes]:
        """Synchronous S3 batch image download."""
        images = {}
        bucket = settings.PROFILE_SEARCH_S3_BUCKET
        for path in image_paths:
            if not path:
                continue
            try:
                resp = self.s3_client.get_object(Bucket=bucket, Key=path)
                images[path] = resp['Body'].read()
            except Exception as e:
                logger.debug(f"Failed to download {path}: {e}")
        return images

    async def close(self):
        """Close all clients."""
        try:
            if self.mongo_client:
                self.mongo_client.close()
            self._executor.shutdown(wait=False)
            logger.info("Profile search service closed")
        except Exception as e:
            logger.debug(f"Profile search close error: {e}")

    async def health_check(self) -> Dict:
        """Return service health status."""
        loop = asyncio.get_running_loop()
        try:
            health = await loop.run_in_executor(self._executor, self._health_sync)
            return health
        except Exception as e:
            return {'connected': False, 'error': str(e)}

    def _health_sync(self) -> Dict:
        """Synchronous health check."""
        result = {
            'connected': True,
            'milvus': False,
            'mongodb': False,
            's3': False,
        }

        try:
            stats = self.milvus_client.get_collection_stats(settings.PROFILE_SEARCH_MILVUS_COLLECTION)
            result['milvus'] = True
            result['milvus_vectors'] = stats.get('row_count', 0)
        except:
            result['connected'] = False

        try:
            self.mongo_client.admin.command('ping')
            result['mongodb'] = True
            result['mongodb_docs'] = self.mongo_collection.estimated_document_count()
        except:
            result['connected'] = False

        if self.s3_client:
            try:
                self.s3_client.head_bucket(Bucket=settings.PROFILE_SEARCH_S3_BUCKET)
                result['s3'] = True
            except:
                pass  # S3 is optional

        return result
```

**Key design decisions:**

1. **Thread pool executor** — pymilvus, pymongo, and boto3 are all synchronous libraries. We run them in a thread pool (4 workers) to avoid blocking the async event loop. Same pattern as `chroma_service.py` uses with `run_in_executor`.

2. **Batch operations** — MongoDB lookup uses `$in` query for all entity_ids at once (single round-trip). S3 downloads are sequential per image but run in the executor thread.

3. **Optional S3 images** — `include_image=False` by default. Profile images are only downloaded when the caller specifically requests them (e.g., for the frontend results endpoint). During pipeline processing, we just save the image path.

4. **Graceful degradation** — All errors return empty list. Profile search never blocks video processing.

---

### Step 3: State (`app/services/state.py`)

Add after `redis_service` (~line 26):

```python
# Profile search service (direct Milvus + MongoDB + S3)
profile_search_service = None  # Optional[ProfileSearchService]
```

---

### Step 4: Lifespan (`core/server.py`)

Add as service #6, after DOSASHOP (~line 98), before the background cleanup task:

```python
# 6. Profile Search (Direct Milvus + MongoDB + S3)
if settings.PROFILE_SEARCH_ENABLED:
    try:
        from app.services.profile_search_service import ProfileSearchService
        state.profile_search_service = ProfileSearchService()
        ps_ok = await state.profile_search_service.connect()
        if ps_ok:
            logger.info("Profile search connected (Milvus + MongoDB + S3)")
        else:
            logger.warning("Profile search connection failed — disabled")
            state.profile_search_service = None
    except Exception as e:
        logger.error(f"Profile search init failed: {e}")
        state.profile_search_service = None
else:
    logger.info("Profile search disabled (PROFILE_SEARCH_ENABLED=false)")
```

Add to the shutdown section (~line 120):

```python
if state.profile_search_service:
    await state.profile_search_service.close()
```

---

### Step 5: Database Schema (`app/database/connection.py`)

Add to `SCHEMA_SQL` after the `identity_links` table (~line 244):

```sql
-- Profile matches (social media profiles matched to video identities)
CREATE TABLE IF NOT EXISTS profile_matches (
    match_id SERIAL PRIMARY KEY,
    job_id VARCHAR(36) NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    tenant_id VARCHAR(100) DEFAULT 'default',
    cluster_id INTEGER NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    similarity FLOAT NOT NULL,
    platform VARCHAR(50) NOT NULL,
    profile_name VARCHAR(255),
    profile_id VARCHAR(255),
    profile_image_path TEXT,
    profile_url TEXT,
    raw_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(job_id, cluster_id, entity_id, platform)
);

CREATE INDEX IF NOT EXISTS idx_profile_matches_job ON profile_matches(job_id);
CREATE INDEX IF NOT EXISTS idx_profile_matches_tenant ON profile_matches(tenant_id);
CREATE INDEX IF NOT EXISTS idx_profile_matches_entity ON profile_matches(entity_id);
```

**Design notes:**
- `profile_image_path` stores the S3 key (e.g., `profile_images/abc123.jpg`), NOT a full URL — URLs are generated at serving time
- `raw_metadata` stores the complete MongoDB metadata object for future use
- UNIQUE constraint on `(job_id, cluster_id, entity_id, platform)` prevents duplicates on reprocessing
- ON DELETE CASCADE — when a job is deleted, its profile matches are cleaned up

---

### Step 6: Database Operations (`app/database/operations.py`)

Add these functions (follows existing `save_identity_link` pattern):

```python
async def save_profile_match(
    job_id: str, cluster_id: int, entity_id: str,
    similarity: float, platform: str,
    tenant_id: str = 'default',
    profile_name: str = None, profile_id: str = None,
    profile_image_path: str = None, profile_url: str = None,
    raw_metadata: dict = None,
) -> int:
    """Save a profile search match. Upserts on conflict."""
    import json
    pool = await get_async_pool()
    match_id = await pool.fetchval(
        """
        INSERT INTO profile_matches (
            job_id, tenant_id, cluster_id, entity_id, similarity,
            platform, profile_name, profile_id, profile_image_path,
            profile_url, raw_metadata
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (job_id, cluster_id, entity_id, platform)
        DO UPDATE SET
            similarity = EXCLUDED.similarity,
            profile_name = EXCLUDED.profile_name,
            profile_image_path = EXCLUDED.profile_image_path,
            profile_url = EXCLUDED.profile_url,
            raw_metadata = EXCLUDED.raw_metadata
        RETURNING match_id
        """,
        job_id, tenant_id, cluster_id, entity_id, similarity,
        platform, profile_name, profile_id, profile_image_path,
        profile_url, json.dumps(raw_metadata) if raw_metadata else None
    )
    return match_id


async def get_profile_matches(
    job_id: str, cluster_id: int = None, tenant_id: str = 'default'
) -> list:
    """Get profile matches for a job, optionally filtered by cluster."""
    pool = await get_async_pool()
    if cluster_id is not None:
        rows = await pool.fetch(
            """
            SELECT * FROM profile_matches
            WHERE tenant_id = $1 AND job_id = $2 AND cluster_id = $3
            ORDER BY similarity DESC
            """,
            tenant_id, job_id, cluster_id
        )
    else:
        rows = await pool.fetch(
            """
            SELECT * FROM profile_matches
            WHERE tenant_id = $1 AND job_id = $2
            ORDER BY cluster_id, similarity DESC
            """,
            tenant_id, job_id
        )
    return [dict(row) for row in rows]
```

---

### Step 7: Integration in Post-Processing (`app/api/v1/v3_face.py`)

Modify `_store_and_match_identities()` (~line 700). Add the profile search block **after** the ChromaDB section, **inside** the identity loop:

```python
# Inside _store_and_match_identities(), after the ChromaDB cross-video matching block:

        # Profile search (direct Milvus + MongoDB)
        if state.profile_search_service:
            try:
                profiles = await state.profile_search_service.search_profiles(
                    embedding=embedding,
                    include_image=False,  # Don't download images during processing
                )
                if profiles:
                    identity_profiles = []
                    for profile in profiles:
                        await db.save_profile_match(
                            job_id=job_id,
                            cluster_id=cluster_id,
                            entity_id=profile['entity_id'],
                            similarity=profile['similarity'],
                            platform=profile['platform'],
                            tenant_id=tenant_id,
                            profile_name=profile.get('name'),
                            profile_id=profile.get('profile_id'),
                            profile_image_path=profile.get('profile_image_path'),
                            profile_url=profile.get('profile_url'),
                            raw_metadata=profile,
                        )
                        identity_profiles.append({
                            'entity_id': profile['entity_id'],
                            'similarity': profile['similarity'],
                            'platform': profile['platform'],
                            'name': profile.get('name'),
                            'profile_id': profile.get('profile_id'),
                            'profile_url': profile.get('profile_url'),
                            'profile_image_path': profile.get('profile_image_path'),
                        })

                    if identity_profiles:
                        identity['profile_matches'] = identity_profiles
                        logger.info(
                            f"Job {job_id} cluster {cluster_id}: "
                            f"{len(identity_profiles)} profile matches"
                        )

            except Exception as e:
                logger.warning(
                    f"Profile search failed for job {job_id} cluster {cluster_id}: {e}"
                )
```

**Important:** This runs BEFORE `_strip_base64_from_result()` — the embedding is still available in `best_face`.

### Execution Order (Updated)

```
① _store_full_results()              → PostgreSQL
② _upload_all_face_crops_parallel()   → DOSASHOP
③ _save_blob_refs_to_db()            → PostgreSQL (blob refs)
④ _store_and_match_identities()      → ChromaDB + Profile Search  ← MODIFIED
⑤ _strip_base64_from_result()        → Clean dict
⑥ Save to Redis
```

---

### Step 8: New API Endpoint (`app/api/v1/v3_face.py`)

Add a new endpoint for querying profile matches:

```python
@router.get("/v3/identity/{job_id}/{cluster_id}/profiles")
async def get_profile_matches_endpoint(
    job_id: str,
    cluster_id: int,
    include_image: bool = False,
    x_tenant_id: str = Header(default="default"),
):
    """Get social media profile matches for a video identity."""
    if not state.db_pool:
        raise HTTPException(status_code=503, detail="Database not connected")

    try:
        matches = await db.get_profile_matches(
            job_id, cluster_id, tenant_id=x_tenant_id
        )

        # Optionally fetch profile images from S3
        if include_image and state.profile_search_service and matches:
            loop = asyncio.get_running_loop()
            image_paths = [m['profile_image_path'] for m in matches if m.get('profile_image_path')]
            if image_paths:
                image_map = await loop.run_in_executor(
                    None,
                    lambda: state.profile_search_service._download_images(image_paths)
                )
                for match in matches:
                    path = match.get('profile_image_path', '')
                    if path in image_map:
                        img_bytes = image_map[path]
                        ext = 'jpeg' if path.endswith('.jpg') else 'webp'
                        match['profile_image_base64'] = (
                            f"data:image/{ext};base64,"
                            + base64.b64encode(img_bytes).decode()
                        )

        # Serialize datetime
        for match in matches:
            if match.get('created_at'):
                match['created_at'] = match['created_at'].isoformat()

        return {
            "job_id": job_id,
            "tenant_id": x_tenant_id,
            "cluster_id": cluster_id,
            "profile_matches": matches,
            "count": len(matches),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

### Step 9: Health Endpoint Update

Add profile search to the existing `/api/v1/health` response:

```python
# In the health check endpoint
if state.profile_search_service:
    health_status['profile_search'] = await state.profile_search_service.health_check()
else:
    health_status['profile_search'] = {
        'connected': False,
        'enabled': settings.PROFILE_SEARCH_ENABLED,
    }
```

---

### Step 10: K8s Deployment (`draft-deployment/dev/face-ocr-api.yaml`)

#### Environment Variables

Add to the face-ocr-api container spec:

```yaml
- name: PROFILE_SEARCH_ENABLED
  value: "true"
- name: PROFILE_SEARCH_MILVUS_URI
  value: "http://172.31.30.225:19530"
- name: PROFILE_SEARCH_MILVUS_COLLECTION
  value: "buffalo_l"
- name: PROFILE_SEARCH_MONGO_URI
  value: "mongodb://serveradmin:<MONGO_PASSWORD>@<MILVUS_HOST>:27017"
- name: PROFILE_SEARCH_MONGO_DB
  value: "media_recognizer"
- name: PROFILE_SEARCH_MONGO_COLLECTION
  value: "GenericFeatures"
- name: PROFILE_SEARCH_S3_BUCKET
  value: "milvus-storage-2025"
- name: PROFILE_SEARCH_S3_REGION
  value: "us-east-1"
- name: PROFILE_SEARCH_S3_ACCESS_KEY
  value: "<AWS_ACCESS_KEY_ID>"
- name: PROFILE_SEARCH_S3_SECRET_KEY
  value: "<AWS_SECRET_ACCESS_KEY>"
- name: PROFILE_SEARCH_MIN_SIMILARITY
  value: "0.70"
- name: PROFILE_SEARCH_TOP_K
  value: "5"
```

#### WARP VPN Sidecar

The face-ocr-api pod needs a WARP sidecar to reach `172.31.30.225` (Milvus + MongoDB are behind the AWS VPN). Copy the WARP sidecar configuration from `media-recognizer-aws-milvus` deployment:

```yaml
# Add as second container in pod spec
- name: warp-ubuntu
  image: <warp-image>  # Same image as media-recognizer-aws-milvus uses
  # ... copy full WARP config from media-recognizer-aws-milvus
```

Check the existing WARP sidecar config:
```bash
kubectl get deployment media-recognizer-aws-milvus -n default -o yaml | grep -A 50 "warp-ubuntu"
```

---

## Response Format

### In Video Results (`GET /v3/results/{job_id}`)

Each identity now includes `profile_matches`:

```json
{
  "identities": [
    {
      "cluster_id": 0,
      "face_count": 45,
      "avg_quality": 0.72,
      "best_face": { ... },
      "profile_matches": [
        {
          "entity_id": "089cac6fdc9b1267...",
          "similarity": 0.7693,
          "platform": "twitter",
          "name": "Pericles 'Perry' Abbasi",
          "profile_id": "2359971372",
          "profile_url": "https://twitter.com/ElectionLegal",
          "profile_image_path": "profile_images/089cac6f...jpg"
        }
      ]
    }
  ]
}
```

### Profile Matches Endpoint (`GET /v3/identity/{job_id}/{cluster_id}/profiles`)

```json
{
  "job_id": "abc-123",
  "tenant_id": "default",
  "cluster_id": 0,
  "count": 3,
  "profile_matches": [
    {
      "match_id": 1,
      "entity_id": "089cac6fdc9b1267...",
      "similarity": 0.7693,
      "platform": "twitter",
      "profile_name": "Pericles 'Perry' Abbasi",
      "profile_id": "2359971372",
      "profile_url": "https://twitter.com/ElectionLegal",
      "profile_image_path": "profile_images/089cac6f...jpg",
      "profile_image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
      "created_at": "2026-02-22T16:00:00Z"
    }
  ]
}
```

Use `?include_image=true` to get base64-encoded profile photos in the response.

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Milvus down at startup | `connect()` returns False, service set to None, profile search disabled |
| MongoDB down at startup | `connect()` returns False, service set to None |
| Milvus timeout during search | `search_profiles()` catches exception, returns `[]`, logs warning |
| MongoDB timeout during lookup | Same — returns `[]` |
| S3 credentials missing | S3 client not created, images unavailable, metadata still works |
| S3 image download fails | Individual image skipped, other results still returned |
| No matches above threshold | Returns `[]` (normal) |
| Outlier identity (cluster_id=-1) | Skipped — never sent to profile search |
| Missing embedding on identity | Skipped — `continue` to next identity |
| Duplicate video reprocessing | UNIQUE constraint upserts instead of duplicating |

Profile search is always **optional and non-blocking**. A failure never prevents video processing from completing.

---

## Testing Plan

### 1. Verify Connectivity (Before Implementation)

From the WARP container:

```bash
# Already tested Feb 22 — all 3 work:
kubectl exec <warp-pod> -- python3 -c "
from pymilvus import MilvusClient
client = MilvusClient(uri='http://172.31.30.225:19530')
print('Milvus OK:', client.get_collection_stats('buffalo_l'))
"

kubectl exec <warp-pod> -- python3 -c "
from pymongo import MongoClient
client = MongoClient('mongodb://serveradmin:<MONGO_PASSWORD>@<MILVUS_HOST>:27017')
print('MongoDB OK:', client.media_recognizer.GenericFeatures.estimated_document_count())
"

kubectl exec <warp-pod> -- python3 -c "
import boto3
s3 = boto3.client('s3', aws_access_key_id='<AWS_ACCESS_KEY_ID>',
    aws_secret_access_key='<AWS_SECRET_ACCESS_KEY>', region_name='us-east-1')
print('S3 OK:', s3.head_bucket(Bucket='milvus-storage-2025'))
"
```

### 2. Local Testing

```bash
# Start face-ocr-api with profile search enabled
# (Need WARP/VPN access to 172.31.30.225 from local machine)
export PROFILE_SEARCH_ENABLED=true
export PROFILE_SEARCH_MILVUS_URI=http://172.31.30.225:19530
export PROFILE_SEARCH_MONGO_URI=mongodb://serveradmin:<MONGO_PASSWORD>@<MILVUS_HOST>:27017
export PROFILE_SEARCH_S3_ACCESS_KEY=<AWS_ACCESS_KEY_ID>
export PROFILE_SEARCH_S3_SECRET_KEY=<AWS_SECRET_ACCESS_KEY>

USE_RAY_CLUSTER=true CHROMA_ENABLED=true \
python -m uvicorn backend.local_server:app --host 0.0.0.0 --port 8002
```

### 3. End-to-End Test

```bash
# Process a video
curl -X POST http://localhost:8002/v3/process -F "file=@test.mp4" -F "mode=face"

# Poll until complete
curl http://localhost:8002/v3/status/{job_id}

# Check results — look for profile_matches
curl http://localhost:8002/v3/results/{job_id} | python -m json.tool | grep -A5 profile_matches

# Query profiles directly
curl "http://localhost:8002/v3/identity/{job_id}/0/profiles?include_image=true"
```

### 4. Database Verification

```sql
SELECT job_id, cluster_id, entity_id, similarity, platform, profile_name
FROM profile_matches
ORDER BY created_at DESC
LIMIT 20;
```

---

## Performance Expectations

| Operation | Time | Notes |
|-----------|------|-------|
| Milvus search (per identity) | ~400ms | Consistent, mmap-backed |
| Milvus batch (5 at once) | ~400ms total | 80ms per query |
| MongoDB metadata lookup | ~1.7s | Batch $in query for all entity_ids |
| S3 image download (5 images) | ~2.2s | Sequential, 20-40KB each |
| **Total per identity (no images)** | **~2.1s** | |
| **Total per identity (with images)** | **~4.3s** | |

For a video with 10 identities: ~21s for profile search (without images). This runs in the post-processing phase which already takes 5-10s for ChromaDB + DOSASHOP, so total post-processing becomes ~30s.

---

## Service Dependencies (Updated)

After integration, face-ocr-api connects to **6 services** (3 new for profile search):

| # | Service | Purpose | Required? |
|---|---------|---------|-----------|
| 1 | PostgreSQL | Persistent storage | Yes |
| 2 | Redis | Job cache + status | Yes |
| 3 | Ray Cluster | GPU inference | Yes |
| 4 | ChromaDB | Cross-video face matching | Optional |
| 5 | DOSASHOP | Azure Blob Storage for face crops | Optional |
| 6 | **Milvus** | **Profile vector search (1M+ faces)** | **Optional (new)** |
| 7 | **MongoDB** | **Profile metadata (name, platform, URL)** | **Optional (new)** |
| 8 | **S3** | **Profile photos** | **Optional (new)** |

Services 6-8 are wrapped by `ProfileSearchService` and enabled/disabled together via `PROFILE_SEARCH_ENABLED`.

---

*End of Document*
