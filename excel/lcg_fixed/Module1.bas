Attribute VB_Name = "Module1"
'===============================================================================
' Test Module for clsPortableLCG
'===============================================================================
' Run these tests in VBA and compare output with the Python test_portable_lcg.py
' All values should match exactly.
'===============================================================================

Option Explicit



Sub TestPortableLCG()
    '
    ' Main test routine - runs all LCG tests
    '
    Debug.Print String(80, "=")
    Debug.Print "PORTABLE LCG VALIDATION TEST (VBA)"
    Debug.Print String(80, "=")
    Debug.Print ""

    ' Run all tests
    Call TestLCGSequence(1, 20)
    Call TestLCGSequence(42, 20)
    Call TestLCGSequence(12345, 20)
    Call TestRandomInt
    Call TestReset

    Debug.Print ""
    Debug.Print String(80, "=")
    Debug.Print "VALIDATION COMPLETE"
    Debug.Print String(80, "=")
    Debug.Print ""
    Debug.Print "Compare this output with Python test_portable_lcg.py"
    Debug.Print "All values should match exactly."
End Sub

Private Sub TestLCGSequence(ByVal Seed As Long, ByVal Count As Long)
    '
    ' Generate a sequence of random numbers for verification
    '
    Dim lcg As New clsPortableLCG
    Dim i As Long
    Dim value As Double

    Debug.Print ""
    Debug.Print String(80, "=")
    Debug.Print "PortableLCG Test - Seed: " & Seed
    Debug.Print String(80, "=")
    Debug.Print "Index" & Space(3) & "State" & Space(10) & "Random()"
    Debug.Print String(80, "-")

    lcg.Initialize Seed

    For i = 1 To Count
        value = lcg.Random()
        Debug.Print Format(i, "0") & Space(8 - Len(CStr(i))) & _
                    Format(lcg.State, "0") & Space(15 - Len(Format(lcg.State, "0"))) & _
                    Format(value, "0.00000000000000000")
    Next i

    Debug.Print String(80, "=")
    Debug.Print ""
End Sub

Private Sub TestRandomInt()
    '
    ' Test the RandomInt() method
    '
    Dim lcg As New clsPortableLCG
    Dim i As Long
    Dim value As Long

    Debug.Print ""
    Debug.Print String(80, "=")
    Debug.Print "PortableLCG RandomInt Test - Seed: 42"
    Debug.Print String(80, "=")
    Debug.Print ""
    Debug.Print "Testing RandomInt(1, 7) - simulating dice roll:"

    lcg.Initialize 42

    For i = 1 To 20
        value = lcg.RandomInt(1, 7)
        Debug.Print "  Roll " & i & ": " & value
    Next i

    Debug.Print String(80, "=")
    Debug.Print ""
End Sub

Private Sub TestReset()
    '
    ' Test that Reset() works correctly
    '
    Dim lcg As New clsPortableLCG
    Dim i As Long
    Dim seq1(1 To 5) As Double
    Dim seq2(1 To 5) As Double
    Dim seq3(1 To 5) As Double
    Dim Match As Boolean

    Debug.Print ""
    Debug.Print String(80, "=")
    Debug.Print "PortableLCG Reset Test"
    Debug.Print String(80, "=")

    lcg.Initialize 42

    Debug.Print ""
    Debug.Print "First sequence (5 values):"
    For i = 1 To 5
        seq1(i) = lcg.Random()
        Debug.Print "  " & i & ": " & Format(seq1(i), "0.00000000000000000")
    Next i

    Debug.Print ""
    Debug.Print "Continuing (5 more values):"
    For i = 1 To 5
        seq2(i) = lcg.Random()
        Debug.Print "  " & (i + 5) & ": " & Format(seq2(i), "0.00000000000000000")
    Next i

    Debug.Print ""
    Debug.Print "Resetting and generating first 5 again:"
    lcg.Reset
    For i = 1 To 5
        seq3(i) = lcg.Random()
        Debug.Print "  " & i & ": " & Format(seq3(i), "0.00000000000000000")
    Next i

    ' Check if sequences match
    Match = True
    For i = 1 To 5
        If seq1(i) <> seq3(i) Then
            Match = False
            Exit For
        End If
    Next i

    Debug.Print ""
    If Match Then
        Debug.Print "SUCCESS: Reset works correctly - sequences match!"
    Else
        Debug.Print "ERROR: Reset failed - sequences don't match!"
    End If

    Debug.Print String(80, "=")
    Debug.Print ""
End Sub

'===============================================================================
' Quick Reference Test - First 10 values with Seed 42
'===============================================================================
Sub QuickTest()
    '
    ' Quick test to verify LCG is working - prints first 10 values with seed 42
    '
    Dim lcg As New clsPortableLCG
    Dim i As Long

    Debug.Print "Quick Test - First 10 random values (Seed=42):"
    lcg.Initialize 42

    For i = 1 To 10
        Debug.Print i & ": " & Format(lcg.Random(), "0.00000000000000000")
    Next i
End Sub

Sub TestNormalDistSync()
    Call InitializePortableRNG(42)

    Debug.Print "Test 3: Integer values (as used in appliances)"
    Debug.Print String(40, "-")

    Dim i As Integer
    For i = 1 To 5
        Debug.Print "  " & i & ": " & GetPortableNormalInteger(1000, 100)
    Next i
End Sub


Sub TestNormInvEdgeCases()
      Dim result As Double

      ' This should work:
      result = Application.WorksheetFunction.NormInv(0.5, 100, 10)
      Debug.Print "Test 1 passed: " & result

      ' This will FAIL (SD = 0):
      On Error Resume Next
      result = Application.WorksheetFunction.NormInv(0.5, 100, 0)
      If Err.Number <> 0 Then
          Debug.Print "Test 2 FAILED as expected: SD=0 is invalid"
          Err.Clear
      End If
      On Error GoTo 0
  End Sub

