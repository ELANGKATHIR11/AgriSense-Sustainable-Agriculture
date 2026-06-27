# Patent Novelty Claim #06: Illiteracy-Aware AI Interface for Agricultural Decision Support

## Title
Method and System for Illiteracy-Aware AI Interface with Icon-First UX, Traffic-Light Severity Indication, and Speech Synthesis for Agricultural Decision Support

## Mechanism

The system implements a three-layer accessibility framework designed for users with limited literacy:

### Layer 1: Icon-First Navigation
- Large (80×80px) icon buttons replace text labels for primary actions:
  - 📷 Camera icon → capture crop image for analysis
  - 🔊 Volume icon → hear instructions spoken aloud
  - 💧 Soil/water icon → connect sensor or check irrigation
  - 🌾 Wheat icon → crop recommendations
- Each icon is a minimum of 20mm touch target on mobile screens
- Icons are universally recognizable crop/agriculture symbols
- Text labels are secondary (10px), serving as accessibility fallback

### Layer 2: Traffic-Light Severity Indication
The Decision Governor's output is visualized as a traffic light:
- 🔴 Red circle (active when action=ACT): "Needs attention!"
- 🟡 Yellow circle (active when action=WAIT or OBSERVE): "Monitoring..."
- 🟢 Green circle (active when action=DO_NOTHING): "All good!"

The traffic light is universally understood regardless of literacy level, language, or cultural context. The active light is fully saturated; inactive lights are desaturated (20% opacity).

### Layer 3: Speech Synthesis Integration
- Every result card has a "🔊 Read Aloud" button using Web Speech API (`window.speechSynthesis`)
- Speech rate is set to 0.85 (85% of normal) for clarity
- Emoji and symbols are stripped before speech synthesis
- Works offline after initial voice data download on most mobile browsers
- Language is matched to the user's selected language preference

### Confidence Band Visualization
Instead of numeric percentages, confidence is shown as a horizontal bar:
- The full bar represents 0-100% range
- A colored band shows the confidence interval (lower to upper bound)
- A dark marker shows the median confidence
- No numbers are required to understand "wide band = uncertain, narrow band = certain"

## Why Non-Obvious

1. **Agricultural AI assumes literacy**: All existing agricultural AI systems present results as text — disease names, treatment instructions, confidence percentages. This system's icon-first design is non-obvious because it reverses the text/visual priority, making text a secondary accessibility feature rather than the primary interface.

2. **Traffic-light for AI confidence**: Using a traffic-light metaphor for AI decision states is non-obvious because:
   - The mapping (ACT→red, DO_NOTHING→green) inverts the typical "green=go, red=stop" metaphor intentionally — red means "your crop needs help" which is the critical state requiring attention
   - The four-state Governor decision is mapped to three lights, with WAIT and OBSERVE sharing yellow, which is a design decision based on farmer behavior studies

3. **Embedded speech synthesis**: Integrating Web Speech API directly into result cards (not as a separate screen-reader) is non-obvious because screen readers read entire pages, while this system speaks only specific agricultural recommendations at the user's request.

4. **Confidence visualization without numbers**: The horizontal confidence band bar communicates calibrated uncertainty to illiterate users — a concept that is typically conveyed through numeric percentages and statistical notation.

## System Claim

A computer-implemented accessibility system for agricultural AI comprising:
- An icon-first navigation layer using universally recognizable agricultural symbols as primary interface elements
- A traffic-light severity indicator mapping Decision Governor output states to a three-light display
- An embedded speech synthesis module using Web Speech API for on-demand reading of specific result components
- A confidence band visualization displaying calibrated uncertainty as a horizontal bar without numeric labels
- Language-adaptive output matching the user's selected language preference

## Method Claim

A method for providing illiteracy-aware agricultural AI decision support comprising:
1. Displaying primary navigation actions as large icon buttons (minimum 20mm touch target) with agriculture-specific symbols
2. Converting Decision Governor output (ACT, WAIT, OBSERVE, DO_NOTHING) to a traffic-light indicator with active/inactive light states
3. Providing a per-result "Read Aloud" button that synthesizes specific agricultural recommendations via Web Speech API at reduced rate (0.85x)
4. Visualizing confidence intervals as horizontal bars where band width indicates certainty level
5. Stripping emoji and non-speech symbols before speech synthesis
6. Operating speech synthesis in the user's selected language

## Dependent Claims

1. The system of the main claim wherein icon buttons auto-arrange based on screen size, maintaining minimum touch target size on mobile devices.
2. The method of the main claim wherein the traffic-light indicator uses haptic feedback (vibration) on mobile devices for the red (ACT) state.
3. The system of the main claim wherein speech synthesis works offline on devices with pre-downloaded voice data.
4. The method of the main claim wherein first-time users receive an audio tour of the interface triggered by a help icon, explaining each section's purpose in spoken language.
