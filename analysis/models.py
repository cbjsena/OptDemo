from django.db import models

class Variable(models.Model):
    """최적화 모델의 결정 변수를 저장하는 테이블"""
    # 어떤 실행에 속한 변수인지 구분하기 위한 ID
    id = models.AutoField(primary_key=True)
    run_id = models.CharField(max_length=50, db_index=True)
    var_name = models.CharField(max_length=255, db_index=True)
    var_group = models.CharField(max_length=100, blank=True, null=True, help_text="변수의 논리적 그룹 (예: plays, is_at_loc)")
    var_type = models.CharField(max_length=20, help_text="BINARY, INTEGER, CONTINUOUS")
    lower_bound = models.FloatField(default=0.0)
    upper_bound = models.FloatField(default=1.0)
    result_value = models.FloatField(blank=True, null=True, help_text="최적화 후 변수의 값")
    slot = models.IntegerField(blank=True, null=True)
    home_team = models.CharField(max_length=100, blank=True, null=True)
    away_team = models.CharField(max_length=100, blank=True, null=True)
    team = models.CharField(max_length=100, blank=True, null=True) # is_at_loc, team_travel 등에서 사용
    city = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        # [신규] run_id와 var_name의 조합이 고유하도록 설정합니다.
        unique_together = ('run_id', 'var_name')

    def __str__(self):
        return self.var_name

class Equation(models.Model):
    """최적화 모델의 제약 조건을 저장하는 테이블"""
    id = models.AutoField(primary_key=True)
    run_id = models.CharField(max_length=50, db_index=True)
    eq_name = models.CharField(max_length=255, db_index=True)
    eq_group = models.CharField(max_length=100, blank=True, null=True, help_text="제약의 논리적 그룹 (예: one_game_per_slot)")
    eq_type = models.CharField(max_length=50, blank=True, null=True, help_text="제약 유형 (예: Linear, Indicator)")
    sign = models.CharField(max_length=2, help_text="<=, >=, ==")
    rhs = models.FloatField(default=0.0, help_text="Right-Hand Side (우변) 값")
    slot = models.IntegerField(blank=True, null=True)
    home_team = models.CharField(max_length=100, blank=True, null=True)
    away_team = models.CharField(max_length=100, blank=True, null=True)
    team = models.CharField(max_length=100, blank=True, null=True)
    city = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        # [신규] run_id와 eq_name의 조합이 고유하도록 설정합니다.
        unique_together = ('run_id', 'eq_name')

    def __str__(self):
        return self.eq_name

class MatrixEntry(models.Model):
    """변수와 제약 조건의 관계(계수)를 저장하는 테이블"""
    run_id = models.CharField(max_length=50, db_index=True)
    variable = models.ForeignKey(Variable, on_delete=models.CASCADE)
    equation = models.ForeignKey(Equation, on_delete=models.CASCADE)
    coefficient = models.FloatField()

    class Meta:
        # [수정] ForeignKey가 변경되었으므로, 이 제약도 업데이트됩니다.
        unique_together = ('variable', 'equation')