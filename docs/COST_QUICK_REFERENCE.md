# InvestLLM Quick Cost Reference

## 🎯 My Recommendation for You

Based on your ₹10-15L budget over 12-18 months:

### Best Setup: Hybrid (Local GPU + Cloud Burst)

```
ONE-TIME INVESTMENT: ₹2,00,000
├── RTX 4090 GPU: ₹1,80,000
├── 1TB NVMe SSD: ₹5,000
├── 32GB RAM (if needed): ₹10,000
└── UPS 1KVA: ₹5,000

MONTHLY RECURRING: ₹15,000 - 25,000
├── Cloud GPU (A100 burst): ₹5,000-10,000
├── Zerodha Kite API: ₹2,000
├── VPS/Infrastructure: ₹5,000-10,000
└── Electricity: ₹3,000
```

---

## 📊 12-Month Cost Breakdown

| Month | Phase | Spend | Cumulative |
|-------|-------|-------|------------|
| 1-2 | Data Collection | ₹15,000 | ₹15,000 |
| 3-4 | Buy GPU + Sentiment | ₹2,35,000 | ₹2,50,000 |
| 5-7 | Price Prediction | ₹50,000 | ₹3,00,000 |
| 8-10 | Strategy Engine | ₹60,000 | ₹3,60,000 |
| 11-12 | Production | ₹90,000 | ₹4,50,000 |

**Year 1 Total: ₹4,50,000** (within your budget!)

---

## 🖥️ GPU Recommendations

### For Development (MUST HAVE)

| GPU | Price | Can Run | Verdict |
|-----|-------|---------|---------|
| **RTX 4090** | ₹1,80,000 | 7B-13B models | ⭐ BEST VALUE |
| RTX 4080 | ₹1,20,000 | 7B models only | OK |
| RTX 3090 (used) | ₹90,000 | 7B-13B models | Budget option |

### For Production (Cloud)

| Provider | GPU | Cost | Use |
|----------|-----|------|-----|
| **RunPod** | RTX 4090 | ₹35/hr | Daily inference |
| **RunPod** | A100 40GB | ₹100/hr | Large training |
| **Vast.ai** | A100 40GB | ₹80/hr | Cheapest A100 |
| **Modal** | A100 | Per-second | Serverless |

---

## 💰 API Costs (Monthly)

### Essential (Can't Skip)

| API | Cost | Purpose |
|-----|------|---------|
| **Zerodha Kite** | ₹2,000 | Real-time + minute data |
| **Total Essential** | **₹2,000** | |

### Recommended

| API | Cost | Purpose |
|-----|------|---------|
| Zerodha Kite | ₹2,000 | Market data |
| Cloud LLM (fallback) | ₹3,000 | Gemini/Claude API |
| Monitoring (Sentry) | ₹2,000 | Error tracking |
| **Total Recommended** | **₹7,000** | |

### FREE APIs (Use These!)

| API | Purpose | Limit |
|-----|---------|-------|
| **yfinance** | Historical prices | Unlimited |
| **Google Gemini** | LLM inference | 60 req/min |
| **HuggingFace** | Model hosting | Unlimited |
| **Cloudflare** | CDN, DNS | Generous |
| **Grafana Cloud** | Monitoring | 10K series |

---

## 🏗️ Infrastructure Costs

### Development Phase (Month 1-6)

```
Local Docker (FREE):
├── PostgreSQL + TimescaleDB
├── Redis
├── Qdrant
└── MLflow

Cloud (Optional):
├── Small VPS: ₹2,000/mo
└── Domain: ₹1,000/year
```

### Production Phase (Month 7+)

| Service | Provider | Cost/Month |
|---------|----------|------------|
| API Server | Hetzner CPX41 | ₹5,000 |
| GPU Inference | RunPod Reserved | ₹15,000 |
| Database | DigitalOcean | ₹3,000 |
| Redis | DigitalOcean | ₹1,500 |
| Monitoring | Grafana Cloud | FREE |
| CDN | Cloudflare Pro | ₹1,500 |
| **Total** | | **₹26,000** |

---

## 📈 Scaling Costs

### Users vs Infrastructure

| Daily Users | GPU Instances | Infra Cost |
|-------------|---------------|------------|
| 1-100 | 1x RTX 4090 | ₹26,000/mo |
| 100-500 | 2x RTX 4090 | ₹45,000/mo |
| 500-1000 | 1x A100 | ₹60,000/mo |
| 1000+ | Multiple A100s | ₹1,00,000+/mo |

---

## ✅ Recommended Purchase Timeline

### Week 1 (Now)
- [ ] 1TB NVMe SSD: ₹5,000

### Month 3-4 (After validating approach)
- [ ] RTX 4090: ₹1,80,000
- [ ] UPS 1KVA: ₹5,000

### Month 6+ (If scaling)
- [ ] Second RTX 4090: ₹1,80,000
- [ ] OR Cloud Reserved Instance

---

## 🔥 Cost Optimization Tips

1. **Use FinGPT**: Saves ₹1.5L in training costs
2. **Start with cloud**: Buy GPU after 3 months
3. **Use Gemini FREE tier**: 60 requests/min is enough for dev
4. **Cache everything**: Redis saves API costs
5. **Use Hetzner**: 50% cheaper than AWS
6. **Quantize models**: 4-bit runs on smaller GPUs
7. **Off-peak training**: Night rates are cheaper

---

## 📱 Quick Decision Guide

### "Should I buy a GPU?"

```
IF you'll use it > 50 hours/month: YES, buy RTX 4090
IF you'll use it < 50 hours/month: NO, use cloud
IF you want flexibility: Hybrid (local + cloud burst)
```

### "Which cloud provider?"

```
For RTX 4090: RunPod or Vast.ai
For A100: Vast.ai (cheapest) or Lambda Labs (reliable)
For serverless: Modal
```

### "How much should I budget monthly?"

```
Learning:     ₹5,000/mo
Development:  ₹15,000/mo
Testing:      ₹30,000/mo
Production:   ₹50,000-60,000/mo
```
