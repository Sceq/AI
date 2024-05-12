from django.db import models
import openai
import os

openai.api_key = os.environ.get('OPENAI_API_KEY')


def generate_description_from_title(title):
    try:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=f"Generowanie opisu na podstawie tytułu: {title}\n",
            temperature=0.7,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Wystąpił problem przy generowaniu opisu: {str(e)}")
        return "{}".format(e)


class TextElement(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField(blank=True)

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        if not self.content:
            self.content = generate_description_from_title(self.title)
        super().save(*args, **kwargs)