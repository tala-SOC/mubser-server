import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:mubser_app/theme/app_theme.dart';
import 'package:mubser_app/screens/home_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  
  // تثبيت اتجاه الشاشة رأسيًا وضبط ألوان شريط النظام
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
      systemNavigationBarColor: Color(0xFF16161E),
      systemNavigationBarIconBrightness: Brightness.light,
    ),
  );

  runApp(const MubserApp());
}

class MubserApp extends StatelessWidget {
  const MubserApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'مُبصر',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.darkTheme,
      // ضبط اتجاه التطبيق من اليمين إلى اليسار للغة العربية
      builder: (context, child) {
        return Directionality(
          textDirection: TextDirection.rtl,
          child: child!,
        );
      },
      home: const HomeScreen(),
    );
  }
}