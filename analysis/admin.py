from django.contrib import admin
from .models import Variable, Equation, MatrixEntry

@admin.register(Variable)
class VariableAdmin(admin.ModelAdmin):
    list_display = ('run_id', 'var_name', 'var_group', 'var_type', 'result_value')
    list_filter = ('run_id', 'var_group', 'var_type')
    search_fields = ('var_name', 'run_id')

@admin.register(Equation)
class EquationAdmin(admin.ModelAdmin):
    list_display = ('run_id', 'eq_name', 'eq_group', 'sign', 'rhs')
    list_filter = ('run_id', 'eq_group', 'sign')
    search_fields = ('eq_name', 'run_id')

@admin.register(MatrixEntry)
class MatrixEntryAdmin(admin.ModelAdmin):
    list_display = ('run_id', 'variable', 'equation', 'coefficient')
    list_filter = ('run_id',)
    search_fields = ('variable__var_name', 'equation__eq_name', 'run_id')