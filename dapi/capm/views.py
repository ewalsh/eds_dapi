from django.shortcuts import render
# from django.http import HttpResponse
from django.views.generic import TemplateView

# Create your views here.


def index(request):
    # return HttpResponse("Initial View")
    return render(request, "base.html")


class MainWrapper(TemplateView):
    template_name = "main.html"


class Index(TemplateView):
    template_name = "base.html"

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super().get_context_data(**kwargs)
        # Add in the new context
        context['user'] = self.request.user
        return context


class Nav(TemplateView):
    template_name = "nav.html"

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super().get_context_data(**kwargs)
        # Add in the new context
        context['name'] = self.request.user
        return context
