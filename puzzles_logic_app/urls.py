from django.urls import path
from . import views

app_name = 'puzzles_logic_app'

urlpatterns = [
    path('', views.main_introduction_view, name='puzzles_logic_introduction'),

    # 1. The Diet Problem
    path('diet-problem/introduction/', views.diet_problem_introduction_view, name='diet_problem_introduction'),
    path('diet-problem/demo/', views.diet_problem_demo_view, name='diet_problem_demo'),

    # 2. Sports Scheduling
    path('sports-scheduling/introduction/', views.sports_scheduling_introduction_view,
         name='sports_scheduling_introduction'),
    path('sports-scheduling/demo/', views.sports_scheduling_demo_view, name='sports_scheduling_demo'),

    # 3. Traveling Salesman Problem (TSP)
    path('tsp/introduction/', views.tsp_introduction_view, name='tsp_introduction'),
    path('tsp/demo/', views.tsp_demo_view, name='tsp_demo'),

    # 4. Sudoku Solver
    path('sudoku/introduction/', views.sudoku_introduction_view, name='sudoku_introduction'),
    path('sudoku/demo/', views.sudoku_demo_view, name='sudoku_demo'),
]