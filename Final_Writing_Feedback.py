import pandas as pd
import random
import re
import matplotlib.pyplot as plt

# Optional seaborn defensively for plotting
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except Exception:
    sns = None
    SEABORN_AVAILABLE = False

# Step 1: Create synthetic student responses for grammar practice
# Common grammar error patterns
error_patterns = {
    'verb_tense': [
        ("I goed to the store yesterday", "I went to the store yesterday", False),
        ("She runned in the park", "She ran in the park", False),
        ("They eated dinner early", "They ate dinner early", False),
        ("I went to the store yesterday", "I went to the store yesterday", True),
        ("She ran in the park", "She ran in the park", True),
    ],
    'subject_verb_agreement': [
        ("He go to school every day", "He goes to school every day", False),
        ("She like ice cream", "She likes ice cream", False),
        ("They goes home now", "They go home now", False),
        ("He goes to school every day", "He goes to school every day", True),
        ("She likes ice cream", "She likes ice cream", True),
    ],
    'plural_forms': [
        ("I have two cat", "I have two cats", False),
        ("There are many book on the shelf", "There are many books on the shelf", False),
        ("Five dog ran past me", "Five dogs ran past me", False),
        ("I have two cats", "I have two cats", True),
        ("There are many books on the shelf", "There are many books on the shelf", True),
    ],
    'articles': [
        ("I saw dog in the park", "I saw a dog in the park", False),
        ("She is teacher", "She is a teacher", False),
        ("He bought apple at store", "He bought an apple at the store", False),
        ("I saw a dog in the park", "I saw a dog in the park", True),
        ("She is a teacher", "She is a teacher", True),
    ],
    'word_order': [
        ("I like very much pizza", "I like pizza very much", False),
        ("She speaks English very well", "She speaks English very well", True),
        ("He always is late", "He is always late", False),
        ("Yesterday I went to school", "Yesterday I went to school", True),
    ]
}

# Generate dataset
data = []
student_id = 1

for error_type, patterns in error_patterns.items():
    # Create multiple students answering each pattern
    for _ in range(10):  # 10 students per pattern set
        pattern = random.choice(patterns)
        student_response, correct_answer, is_correct = pattern

        data.append({
            'student_id': student_id,
            'error_type': error_type,
            'student_response': student_response,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'question_prompt': f"Fix this sentence if needed: {student_response}"
        })
        student_id += 1

# Create DataFrame
df = pd.DataFrame(data)

# Add some metadata
df['response_length'] = df['student_response'].str.len()
df['word_count'] = df['student_response'].str.split().str.len()

# Save to CSV
df.to_csv('student_responses.csv', index=False)

# Display summary
print("Synthetic dataset created!")
print(f"\nTotal responses: {len(df)}")
print(f"Correct responses: {df['is_correct'].sum()}")
print(f"Incorrect responses: {(~df['is_correct']).sum()}")
print("\nError type distribution:")
print(df['error_type'].value_counts())
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset saved as 'student_responses.csv'")



# Step 2: Create rule-based feedback system using what I learned in weeks 4-6
# Optional spaCy support
# If the model is not installed, set SPACY_AVAILABLE to False and nlp to None so
# code will reliably fall back to regex behavior without warnings.
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except Exception:
        nlp = None
        SPACY_AVAILABLE = False
except Exception:
    spacy = None
    nlp = None
    SPACY_AVAILABLE = False


# import language_tool_python to get stronger grammar suggestions
try:
    import language_tool_python
    try:
        # create a tool object (may start a local Java process); defer errors
        _lt_tool = language_tool_python.LanguageTool('en-US')
        LT_AVAILABLE = True
    except Exception:
        _lt_tool = None
        LT_AVAILABLE = False
except Exception:
    language_tool_python = None
    _lt_tool = None
    LT_AVAILABLE = False


def get_languagetool_suggestions(text, max_suggestions=3):
    """Return up to `max_suggestions` human-readable suggestions from LanguageTool.

    If LanguageTool isn't available or fails, return an empty list.
    """
    if not LT_AVAILABLE or _lt_tool is None:
        return []
    try:
        matches = _lt_tool.check(text)
        suggestions = []
        for m in matches:
            # each match has a ruleId, message, replacements
            repls = m.replacements[:2] if m.replacements else []
            suggestion = {
                'message': m.message,
                'offset': m.offset,
                'length': m.errorLength,
                'replacements': repls,
                'context': text[max(0, m.offset-20): m.offset + m.errorLength + 20]
            }
            suggestions.append(suggestion)
            if len(suggestions) >= max_suggestions:
                break
        return suggestions
    except Exception:
        return []


def analyze_grammar(response, use_spacy=False):
    """
    Analyze student response for common grammar errors.

    If use_spacy=True and spaCy + model are available, use spaCy parsing.
    Otherwise, fall back to the original regex/string-based checks so the
    analyzer still works without spaCy.

    Returns: (error_type, feedback_message, is_correct)
    """
    response_lower = response.lower()

    # Use spaCy when requested and available
    if use_spacy and SPACY_AVAILABLE and nlp is not None:
        doc = nlp(response)

        # 1) Check for irregular verb errors using token text
        irregular_verbs = {
            'goed': ('went', 'verb_tense'),
            'runned': ('ran', 'verb_tense'),
            'eated': ('ate', 'verb_tense'),
            'buyed': ('bought', 'verb_tense'),
            'thinked': ('thought', 'verb_tense'),
        }
        for token in doc:
            if token.text.lower() in irregular_verbs:
                correct, error_type = irregular_verbs[token.text.lower()][0], irregular_verbs[token.text.lower()][1]
                return (error_type,
                        f"Verb tense error detected! '{token.text}' should be '{correct}'. "
                        f"Remember: irregular verbs have special past tense forms.",
                        False)

        # 2) Subject-verb agreement using dependency labels and POS tags
        for token in doc:
            if token.dep_ == 'nsubj':
                subj = token
                verb = token.head
                # only consider if head is a verb
                if verb.pos_ == 'VERB' or verb.pos_ == 'AUX':
                    subj_text = subj.text.lower()
                    # third-person singular subjects
                    if subj_text in ('he', 'she', 'it'):
                        # VBZ is the tag for 3rd-person singular present
                        if verb.tag_ != 'VBZ':
                            return ('subject_verb_agreement',
                                    "Subject-verb agreement error! With he/she/it, add 's' or 'es' to the verb.",
                                    False)
                    # non-third-person subjects
                    if subj_text in ('they', 'we', 'you', 'i'):
                        if verb.tag_ == 'VBZ':
                            return ('subject_verb_agreement',
                                    "Subject-verb agreement error! With they/we/you/I, don't add 's' to the verb.",
                                    False)

        # 3) Plural forms: number + singular noun
        for i, token in enumerate(doc):
            if token.like_num or token.pos_ == 'NUM':
                # find the next noun within a short span
                for j in range(i + 1, min(i + 4, len(doc))):
                    nxt = doc[j]
                    if nxt.pos_ == 'NOUN' and nxt.tag_ == 'NN':
                        return ('plural_forms',
                                f"Plural form error! After '{token.text}', the noun should be plural. Try '{nxt.text}s' or the correct plural form.",
                                False)

        # 4) Missing articles: noun without a determiner in contexts like 'saw dog' or 'is teacher'
        for token in doc:
            if token.pos_ == 'NOUN' and token.lemma_.lower() in ['cat', 'dog', 'teacher', 'book', 'car']:
                has_det = any(child.dep_ == 'det' for child in token.children)
                # if there's no determiner and the noun is functioning as object/attr, flag it
                if not has_det and token.dep_ in ('dobj', 'pobj', 'attr', 'nsubj'):
                    return ('articles',
                            f"Missing article! Consider adding 'a' or 'an' before the noun. Example: 'a {token.text}'",
                            False)

        # 5) Word order heuristics
        if 'very much' in response_lower and any(food in response_lower for food in ['pizza', 'ice cream', 'cake']):
            return ('word_order',
                    "Word order issue! In English, 'very much' usually comes after the object. Try: 'I like [object] very much'",
                    False)

        if any(phrase in response_lower for phrase in ['always is', 'never is']):
            return ('word_order',
                    "Word order issue! Adverbs of frequency (always, never) go BEFORE 'is'. Try: 'He is always late' not 'He always is late'",
                    False)

        return (None, "Great job! Your sentence looks grammatically correct!", True)

    # FALLBACK: regex- and string-based checks (works without spaCy)

    # Check for irregular verb errors
    irregular_verbs = {
        'goed': ('went', 'verb_tense'),
        'runned': ('ran', 'verb_tense'),
        'eated': ('ate', 'verb_tense'),
        'buyed': ('bought', 'verb_tense'),
        'thinked': ('thought', 'verb_tense'),
    }

    for wrong, (correct, err_type) in irregular_verbs.items():
        if wrong in response_lower:
            return (err_type,
                    f"Verb tense error detected! '{wrong}' should be '{correct}'. Remember: irregular verbs have special past tense forms.",
                    False)

    # Subject-verb agreement patterns
    patterns = [
        (r'\b(he|she|it)\s+(go|like|want|need|have)\b', 'subject_verb_agreement',
         "Subject-verb agreement error! With he/she/it, add 's' or 'es' to the verb."),
        (r'\b(they|we|you)\s+(goes|likes|wants|needs|has)\b', 'subject_verb_agreement',
         "Subject-verb agreement error! With they/we/you, don't add 's' to the verb."),
    ]

    for pattern, err_type, message in patterns:
        if re.search(pattern, response_lower):
            return (err_type, message, False)

    # Plural form checks using simple number words
    number_words = ['two', 'three', 'four', 'five', 'many', 'several', 'few']
    words = response.split()
    for i, word in enumerate(words):
        if word.lower() in number_words and i + 1 < len(words):
            next_word = words[i + 1]
            if next_word.lower() in ['cat', 'dog', 'book', 'car', 'house', 'person']:
                return ('plural_forms', f"Plural form error! After '{word}', the noun should be plural. Try '{next_word}s'.", False)

    # Missing articles
    article_patterns = [r'\b(saw|is|bought|have)\s+(cat|dog|teacher|book|car)\b']
    for pattern in article_patterns:
        match = re.search(pattern, response_lower)
        if match:
            return ('articles', f"Missing article! Consider adding 'a' or 'an' before the noun. Example: 'a {match.group(2)}'", False)

    # Word order issues
    if 'very much' in response_lower and response_lower.find('very much') < response_lower.rfind(' '):
        if any(food in response_lower for food in ['pizza', 'ice cream', 'cake']):
            return ('word_order', "Word order issue! In English, 'very much' usually comes after the object.", False)

    if 'always is' in response_lower or 'never is' in response_lower:
        return ('word_order', "Word order issue! Adverbs of frequency go BEFORE 'is'.", False)

    # Default: no detected errors
    return (None, "Great job! Your sentence looks grammatically correct!", True)


def generate_personalized_feedback(student_response, error_type=None, use_spacy=True):
    """
    Generate detailed feedback; prefer spaCy but silently fall back to regex if spaCy isn't available.
    Also augment feedback with LanguageTool suggestions when available.
    """
    requested_spacy = bool(use_spacy)
    use_spacy = requested_spacy and SPACY_AVAILABLE and nlp is not None

    if requested_spacy and not use_spacy:
        # Inform user but continue with fallback
        print("Note: spaCy or the 'en_core_web_sm' model is not available; falling back to regex-based analysis.")

    analysis = analyze_grammar(student_response, use_spacy=use_spacy)
    detected_error, feedback, is_correct = analysis

    # Augment feedback with LanguageTool suggestions if available
    lt_suggestions = get_languagetool_suggestions(student_response, max_suggestions=3)
    if lt_suggestions:
        # Build a compact readable summary
        lt_lines = ["LanguageTool suggestions:"]
        for s in lt_suggestions:
            repl_text = ', '.join([str(r) for r in s['replacements']]) if s['replacements'] else 'no replacement suggested'
            lt_lines.append(f"- {s['message']} -> {repl_text}")
        feedback += "\n" + " ; ".join(lt_lines)

    # Add personalized encouragement
    if is_correct:
        encouragement = [
            "Keep up the excellent work!",
            "You're making great progress!",
            "Perfect! You've got this!",
        ]
        import random
        feedback += " " + random.choice(encouragement)
    else:
        tips = {
            'verb_tense': "Tip: Make a list of irregular verbs and practice them daily!",
            'subject_verb_agreement': "Tip: Remember the rule: I/You/We/They + verb, He/She/It + verb+s",
            'plural_forms': "Tip: If there's more than one, the noun usually needs an 's' or 'es'!",
            'articles': "Tip: Use 'a' before consonant sounds, 'an' before vowel sounds.",
            'word_order': "Tip: English usually follows Subject-Verb-Object order.",
        }
        feedback += " " + tips.get(detected_error, "Keep practicing!")

    return {
        'student_response': student_response,
        'error_type': detected_error,
        'feedback': feedback,
        'is_correct': is_correct
    }


# Test with some examples
if __name__ == "__main__":
    test_responses = [
        "I goed to the store yesterday",
        "He go to school every day",
        "I have two cat",
        "I saw dog in the park",
        "She speaks English very well",
        "I like very much pizza",
    ]

    print("Testing Feedback Engine\n")
    print("=" * 60)

    for response in test_responses:
        # Request spaCy-based analysis when possible
        result = generate_personalized_feedback(response, use_spacy=True)
        print(f"\nStudent: '{result['student_response']}'")
        print(f"Error Type: {result['error_type']}")
        print(f"Feedback: {result['feedback']}")
        print("-" * 60)

    # Load and process the synthetic dataset
    print("\n\nProcessing Synthetic Dataset...\n")
    try:
        df = pd.read_csv('student_responses.csv')

        # Apply feedback to all responses
        feedback_results = []
        for _, row in df.iterrows():
            # Request spaCy-based analysis when possible
            result = generate_personalized_feedback(row['student_response'], use_spacy=True)
            feedback_results.append(result)

        # Add feedback to dataframe
        df['generated_feedback'] = [r['feedback'] for r in feedback_results]
        df['detected_correct'] = [r['is_correct'] for r in feedback_results]

        # Save enhanced dataset
        df.to_csv('student_responses_with_feedback.csv', index=False)
        print("Feedback added to all responses!")
        print(f"Enhanced dataset saved as 'student_responses_with_feedback.csv'")

        # Show accuracy
        accuracy = (df['is_correct'] == df['detected_correct']).sum() / len(df) * 100
        print(f"\nFeedback Engine Accuracy: {accuracy:.1f}%")

    except FileNotFoundError:
        print("Run Step 1 first to generate the dataset!")

# Step 3: Analyze the synthetic dataset and student performance
# Uses skills from Week 9-10 (pandas, visualization)

print("Student Performance Analysis\n")
print("=" * 70)

# Load the dataset with feedback
try:
    df = pd.read_csv('student_responses_with_feedback.csv')
    print(f"Loaded {len(df)} student responses\n")
except FileNotFoundError:
    print("Dataset not found. Run Steps 1 and 2 first!")
    exit()

# === BASIC STATISTICS ===
print("\nOVERALL STATISTICS")
print("-" * 70)

total_responses = len(df)
correct_responses = df['is_correct'].sum()
accuracy = (correct_responses / total_responses) * 100

print(f"Total Responses: {total_responses}")
print(f"Correct Responses: {correct_responses}")
print(f"Incorrect Responses: {total_responses - correct_responses}")
print(f"Overall Accuracy: {accuracy:.1f}%")

# === ERROR TYPE ANALYSIS ===
print("\n\nERROR TYPE BREAKDOWN")
print("-" * 70)

error_counts = df['error_type'].value_counts()
print(error_counts)

# Calculate accuracy per error type
print("\n\nAccuracy by Error Type:")
for error_type in df['error_type'].unique():
    error_df = df[df['error_type'] == error_type]
    error_accuracy = (error_df['is_correct'].sum() / len(error_df)) * 100
    print(f"  {error_type}: {error_accuracy:.1f}% correct")

# === RESPONSE LENGTH ANALYSIS ===
print("\n\nRESPONSE LENGTH ANALYSIS")
print("-" * 70)

print(f"Average response length: {df['response_length'].mean():.1f} characters")
print(f"Average word count: {df['word_count'].mean():.1f} words")

# Check if longer responses are more likely to be correct
correct_avg_length = df[df['is_correct']]['response_length'].mean()
incorrect_avg_length = df[~df['is_correct']]['response_length'].mean()
print(f"\nCorrect answers avg length: {correct_avg_length:.1f} characters")
print(f"Incorrect answers avg length: {incorrect_avg_length:.1f} characters")

# === VISUALIZATIONS ===
print("\n\nGenerating visualizations...\n")

# Set style
if SEABORN_AVAILABLE:
    sns.set_style("whitegrid")
else:
    plt.style.use("ggplot")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Student Grammar Practice Analysis', fontsize=16, fontweight='bold')

# 1. Error Type Distribution
ax1 = axes[0, 0]
error_counts.plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Distribution of Error Types')
ax1.set_xlabel('Error Type')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)

# 2. Accuracy by Error Type
ax2 = axes[0, 1]
accuracy_by_type = df.groupby('error_type')['is_correct'].mean() * 100
accuracy_by_type.plot(kind='bar', ax=ax2, color='lightcoral')
ax2.set_title('Accuracy Rate by Error Type')
ax2.set_xlabel('Error Type')
ax2.set_ylabel('Accuracy (%)')
ax2.tick_params(axis='x', rotation=45)
ax2.axhline(y=50, color='red', linestyle='--', label='50% threshold')
ax2.legend()

# 3. Correct vs Incorrect Distribution
ax3 = axes[1, 0]
correct_counts = df['is_correct'].value_counts()
colors = ['lightgreen', 'lightcoral']
ax3.pie(correct_counts, labels=['Correct', 'Incorrect'], autopct='%1.1f%%',
        colors=colors, startangle=90)
ax3.set_title('Overall Correct vs Incorrect')

# 4. Response Length Distribution
ax4 = axes[1, 1]
df[df['is_correct']]['response_length'].hist(ax=ax4, bins=15, alpha=0.6,
                                               label='Correct', color='green')
df[~df['is_correct']]['response_length'].hist(ax=ax4, bins=15, alpha=0.6,
                                                label='Incorrect', color='red')
ax4.set_title('Response Length: Correct vs Incorrect')
ax4.set_xlabel('Response Length (characters)')
ax4.set_ylabel('Frequency')
ax4.legend()

plt.tight_layout()
plt.savefig('student_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'student_analysis.png'")

# === DETAILED ERROR ANALYSIS ===
print("\n\nDETAILED ERROR INSIGHTS")
print("-" * 70)

# Most common words in incorrect responses
print("\nMost common words in incorrect responses:")
incorrect_responses = df[~df['is_correct']]['student_response']
all_words = ' '.join(incorrect_responses).lower().split()

from collections import Counter
word_freq = Counter(all_words)
print(word_freq.most_common(10))

# === RECOMMENDATIONS ===
print("\n\nRECOMMENDATIONS FOR IMPROVEMENT")
print("-" * 70)

# Find the error type with lowest accuracy
worst_error = accuracy_by_type.idxmin()
worst_accuracy = accuracy_by_type.min()

print(f"1. Focus Area: {worst_error}")
print(f"   Current accuracy: {worst_accuracy:.1f}%")
print(f"   This is the most challenging area for students.\n")

print(f"2. Most Common Error: {error_counts.index[0]}")
print(f"   Appears {error_counts.iloc[0]} times in the dataset.\n")

print("3. Suggested Next Steps:")
print("   - Create targeted exercises for weak areas")
print("   - Provide more examples for common errors")
print("   - Add progressive difficulty levels")

# === EXPORT SUMMARY ===
print("\n\nEXPORTING SUMMARY REPORT")
print("-" * 70)

summary_report = {
    'Total Responses': total_responses,
    'Correct': correct_responses,
    'Incorrect': total_responses - correct_responses,
    'Accuracy (%)': accuracy,
    'Most Common Error': error_counts.index[0],
    'Weakest Area': worst_error,
    'Lowest Accuracy (%)': worst_accuracy,
}

summary_df = pd.DataFrame([summary_report])
summary_df.to_csv('analysis_summary.csv', index=False)
print("Summary report saved as 'analysis_summary.csv'")

# Create a detailed error report
error_report = df.groupby('error_type').agg({
    'is_correct': ['count', 'sum', 'mean'],
    'response_length': 'mean',
    'word_count': 'mean'
}).round(2)

error_report.columns = ['Total', 'Correct', 'Accuracy', 'Avg Length', 'Avg Words']
error_report.to_csv('error_type_report.csv')
print("Detailed error report saved as 'error_type_report.csv'")

print("\n" + "=" * 70)
print("Analysis complete! Check the generated files:")
print("   - student_analysis.png")
print("   - analysis_summary.csv")
print("   - error_type_report.csv")
print("=" * 70)

