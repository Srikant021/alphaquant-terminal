
The √7 appears because volatility scales with the square root of time. A week has 7 days, so the weekly move is √7 (about 2.65) times larger than the daily move.

**What the Probabilities Mean:**
- **68% chance** price stays within the daily lines
- **95% chance** price stays within 2× the daily lines
- **99.7% chance** price stays within 3× the daily lines

**Practical Trading Uses:**
1. **Stop Loss Placement:** Put your stop loss just outside the expected range
2. **Option Selling:** Sell options with strike prices beyond the expected range for higher win rates
3. **Breakout Trading:** If price breaks the expected range early in the day, it signals unusual strength/weakness

**Real Example:**
If Nifty is at 23,000 with an expected move of ±150 points:
- Tomorrow's range: 22,850 - 23,150
- You might sell a 23,200 call and 22,800 put, collecting premium
- If price stays within range (68% probability), both options expire worthless
    """,
    
    "Hurst Exponent": """
### 🔬 What is the Hurst Exponent?

The Hurst Exponent tells you whether the market has "memory" - does the past predict the future? It classifies markets into three types.

**The Three Market Types:**

1. **Trending (H > 0.55):**
   - What happened yesterday likely continues today
   - Up days tend to follow up days
   - Down days tend to follow down days
   - Like a ball rolling downhill - it keeps going

2. **Mean-Reverting (H < 0.45):**
   - What goes up must come down
   - After a big up day, expect a down day
   - Price keeps returning to its average
   - Like a rubber band - the more it stretches, the harder it snaps back

3. **Random Walk (H ≈ 0.50):**
   - Each day is independent
   - Yesterday tells you nothing about tomorrow
   - Like flipping a coin - past flips don't affect future flips

**How to Read the Chart:**
- **Top Panel:** Price action
- **Bottom Panel:** 60-day rolling Hurst value
- **Green Line:** Trending threshold (0.55)
- **Red Line:** Mean-reverting threshold (0.45)

**Trading Strategies by Regime:**

| Regime | Strategy | Example |
|--------|----------|---------|
| Trending | Trend Following | Buy pullbacks, trail stops |
| Mean-Reverting | Range Trading | Buy support, sell resistance |
| Random Walk | Non-Directional | Iron Condors, Straddles |

**Real Example:**
If Hurst drops from 0.62 (trending) to 0.41 (mean-reverting):
- Stop using trend-following strategies
- Switch to range-bound strategies
- Take profits at moving averages instead of letting them run
    """,
    
    "IV Rank & IV Percentile": """
### 📉 What are IV Rank and IV Percentile?

These tell you whether options are currently expensive or cheap compared to history. This is crucial because option prices are driven by volatility.

**The Difference:**
- **IV Rank:** Where current volatility sits between the 1-year high and low
- **IV Percentile:** What percentage of days had lower volatility than today

**How to Read the Chart:**
- **Blue Line:** 20-day historical volatility
- **Red Dashed Line:** 1-year maximum volatility
- **Green Dashed Line:** 1-year minimum volatility
- **White Line:** Current volatility level

**Understanding the Numbers:**

| IV Rank | Meaning | Strategy |
|---------|---------|----------|
| 0-30% | Options are CHEAP | Buy options (Debit Spreads, Long Straddles) |
| 30-65% | Options are FAIR | Neutral strategies |
| 65-100% | Options are EXPENSIVE | Sell options (Credit Spreads, Iron Condors) |

**Why This Matters:**
Buying options when they're expensive is like buying insurance during a hurricane - you pay too much. Selling options when they're expensive is like selling umbrellas in the rain - you get premium prices.

**Real Example:**
If IV Rank is 85% and IV Percentile is 92%:
- Options are extremely expensive
-