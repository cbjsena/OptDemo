from django.test import TestCase
from django.urls import reverse


class ComplexAppTests(TestCase):
    def test_complex_intro_page(self):
        response = self.client.get(reverse('complex_app:complex_app_introduction'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'complex_app/complex_app_introduction.html')
        self.assertContains(response, 'Complex Optimization')

    def test_palletizing_intro_page(self):
        response = self.client.get(reverse('complex_app:palletizing_introduction'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'complex_app/palletizing_introduction.html')
        self.assertContains(response, '3D Palletizing Optimization')

    def test_palletizing_demo_get(self):
        response = self.client.get(reverse('complex_app:palletizing_demo'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'complex_app/palletizing_demo.html')
        self.assertContains(response, '3D Palletizing Demo')

    def test_palletizing_demo_post_success(self):
        post_data = {
            'num_types': '2',
            'pallet_l': '120', 'pallet_w': '100', 'pallet_h': '100', 'pallet_max_weight': '500',
            'box_0_id': 'A', 'box_0_l': '40', 'box_0_w': '30', 'box_0_h': '20', 'box_0_weight': '10', 'box_0_qty': '4', 'box_0_rotatable': 'on',
            'box_1_id': 'B', 'box_1_l': '50', 'box_1_w': '40', 'box_1_h': '25', 'box_1_weight': '12', 'box_1_qty': '3', 'box_1_rotatable': 'on',
        }
        response = self.client.post(reverse('complex_app:palletizing_demo'), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.context.get('results'))
        self.assertContains(response, '적재 완료')

    def test_palletizing_demo_post_validation_error(self):
        post_data = {
            'num_types': '1',
            'pallet_l': '0', 'pallet_w': '100', 'pallet_h': '100', 'pallet_max_weight': '500',
            'box_0_id': 'A', 'box_0_l': '40', 'box_0_w': '30', 'box_0_h': '20', 'box_0_weight': '10', 'box_0_qty': '2',
        }
        response = self.client.post(reverse('complex_app:palletizing_demo'), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '입력값 오류')

