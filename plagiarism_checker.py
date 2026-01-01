from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def vectorize_text(texts):
    """
    Converts a list of text documents into TF-IDF vectors.
    """
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts).toarray()


def calculate_similarity(vector1, vector2):
    """
    Calculates cosine similarity between two vectors.
    """
    return cosine_similarity([vector1, vector2])[0][1]


def check_plagiarism(student_files, student_texts):
    """
    Compares each student's text with others and
    returns similarity scores.
    """
    vectors = vectorize_text(student_texts)
    student_vectors = list(zip(student_files, vectors))

    results = set()

    for student_a, vector_a in student_vectors:
        remaining_students = student_vectors.copy()
        remaining_students.remove((student_a, vector_a))

        for student_b, vector_b in remaining_students:
            similarity_score = calculate_similarity(vector_a, vector_b)
            pair = tuple(sorted((student_a, student_b)))
            results.add((pair[0], pair[1], round(similarity_score, 2)))

    return results


if __name__ == "__main__":
    student_files = [
        "student1.txt",
        "student2.txt",
        "student3.txt"
    ]

    student_texts = [
        "Machine learning is a field of artificial intelligence.",
        "Artificial intelligence includes machine learning concepts.",
        "I love playing football on weekends."
    ]

    plagiarism_results = check_plagiarism(student_files, student_texts)

    for result in plagiarism_results:
        print(result)
