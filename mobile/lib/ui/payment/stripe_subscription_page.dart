import 'dart:async';

import 'package:flutter/material.dart';
import "package:logging/logging.dart";
import 'package:photos/ente_theme_data.dart';
import "package:photos/generated/l10n.dart";
import 'package:photos/models/billing_plan.dart';
import 'package:photos/models/subscription.dart';
import 'package:photos/models/user_details.dart';
import 'package:photos/services/billing_service.dart';
import 'package:photos/services/user_service.dart';
import "package:photos/theme/colors.dart";
import 'package:photos/theme/ente_theme.dart';
import 'package:photos/ui/common/loading_widget.dart';
import 'package:photos/ui/common/progress_dialog.dart';
import 'package:photos/ui/common/web_page.dart';
import 'package:photos/ui/components/buttons/button_widget.dart';
import "package:photos/ui/components/captioned_text_widget.dart";
import "package:photos/ui/components/menu_item_widget/menu_item_widget.dart";
import "package:photos/ui/components/title_bar_title_widget.dart";
import 'package:photos/ui/payment/child_subscription_widget.dart';
import 'package:photos/ui/payment/payment_web_page.dart';
import 'package:photos/ui/payment/skip_subscription_widget.dart';
import 'package:photos/ui/payment/subscription_common_widgets.dart';
import 'package:photos/ui/payment/subscription_plan_widget.dart';
import "package:photos/ui/payment/view_add_on_widget.dart";
import "package:photos/utils/data_util.dart";
import 'package:photos/utils/dialog_util.dart';
import 'package:photos/utils/toast_util.dart';
import 'package:step_progress_indicator/step_progress_indicator.dart';
import 'package:url_launcher/url_launcher_string.dart';

class StripeSubscriptionPage extends StatefulWidget {
  final bool isOnboarding;

  const StripeSubscriptionPage({
    this.isOnboarding = false,
    super.key,
  });

  @override
  State<StripeSubscriptionPage> createState() => _StripeSubscriptionPageState();
}

class _StripeSubscriptionPageState extends State<StripeSubscriptionPage> {
  final _billingService = BillingService.instance;
  final _userService = UserService.instance;
  Subscription? _currentSubscription;
  late ProgressDialog _dialog;
  late UserDetails _userDetails;

  // indicates if user's subscription plan is still active
  late bool _hasActiveSubscription;
  bool _hideCurrentPlanSelection = false;
  late FreePlan _freePlan;
  List<BillingPlan> _plans = [];
  bool _hasLoadedData = false;
  bool _isLoading = false;
  bool _isStripeSubscriber = false;
  bool _showYearlyPlan = false;
  EnteColorScheme colorScheme = darkScheme;
  final Logger logger = Logger("StripeSubscriptionPage");

  Future<void> _fetchSub() async {
    return _userService
        .getUserDetailsV2(memoryCount: false)
        .then((userDetails) async {
      _userDetails = userDetails;
      _currentSubscription = userDetails.subscription;

      _showYearlyPlan = _currentSubscription!.isYearlyPlan();
      _hideCurrentPlanSelection =
          (_currentSubscription?.attributes?.isCancelled ?? false) &&
              userDetails.hasPaidAddon();
      _hasActiveSubscription = _currentSubscription!.isValid();
      _isStripeSubscriber = _currentSubscription!.paymentProvider == stripe;

      if (_isStripeSubscriber && _currentSubscription!.isPastDue()) {
        _redirectToPaymentPortal();
      }

      return _filterStripeForUI().then((value) {
        _hasLoadedData = true;
        setState(() {});
      });
    });
  }

  // _filterPlansForUI is used for initializing initState & plan toggle states
  Future<void> _filterStripeForUI() async {
    final billingPlans = await _billingService.getBillingPlans();
    _freePlan = billingPlans.freePlan;
    _plans = billingPlans.plans.where((plan) {
      if (plan.stripeID.isEmpty) {
        return false;
      }
      final isYearlyPlan = plan.period == 'year';
      return isYearlyPlan == _showYearlyPlan;
    }).toList();
    setState(() {});
  }

  FutureOr onWebPaymentGoBack(dynamic value) async {
    // refresh subscription
    await _dialog.show();
    try {
      await _fetchSub();
    } catch (e) {
      showToast(context, "Failed to refresh subscription");
    }
    await _dialog.hide();

    // verify user has subscribed before redirecting to main page
    if (widget.isOnboarding &&
        _currentSubscription != null &&
        _currentSubscription!.isValid() &&
        _currentSubscription!.productID != freeProductID) {
      Navigator.of(context).popUntil((route) => route.isFirst);
    }
  }

  @override
  Widget build(BuildContext context) {
    colorScheme = getEnteColorScheme(context);
    final textTheme = getEnteTextTheme(context);

    return Scaffold(
      appBar: widget.isOnboarding
          ? AppBar(
              scrolledUnderElevation: 0,
              elevation: 0,
              title: Hero(
                tag: "subscription",
                child: StepProgressIndicator(
                  totalSteps: 4,
                  currentStep: 4,
                  selectedColor: Theme.of(context).colorScheme.greenAlternative,
                  roundedEdges: const Radius.circular(10),
                  unselectedColor:
                      Theme.of(context).colorScheme.stepProgressUnselectedColor,
                ),
              ),
            )
          : AppBar(
              scrolledUnderElevation: 0,
              toolbarHeight: 48,
              leadingWidth: 48,
              leading: GestureDetector(
                onTap: () {
                  Navigator.pop(context);
                },
                child: const Icon(
                  Icons.arrow_back_outlined,
                ),
              ),
            ),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TitleBarTitleWidget(
                  title:
                      widget.isOnboarding ? "Select your plan" : "Subscription",
                ),
                _isFreePlanUser() || !_hasLoadedData
                    ? const SizedBox.shrink()
                    : Text(
                        convertBytesToReadableFormat(
                          _userDetails.getTotalStorage(),
                        ),
                        style: textTheme.smallMuted,
                      ),
              ],
            ),
          ),
          Expanded(child: _getBody()),
        ],
      ),
    );
  }

  Widget _getBody() {
    if (!_isLoading) {
      _isLoading = true;
      _dialog = createProgressDialog(context, S.of(context).pleaseWait);
      _fetchSub();
    }
    if (_hasLoadedData) {
      if (_userDetails.isPartOfFamily() && !_userDetails.isFamilyAdmin()) {
        return ChildSubscriptionWidget(userDetails: _userDetails);
      } else {
        return _buildPlans();
      }
    }
    return const EnteLoadingWidget();
  }

  Widget _buildPlans() {
    final widgets = <Widget>[];

    widgets.add(
      SubscriptionHeaderWidget(
        isOnboarding: widget.isOnboarding,
        currentUsage: _userDetails.getFamilyOrPersonalUsage(),
      ),
    );

    widgets.add(_showSubscriptionToggle());

    widgets.addAll([
      Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: _getStripePlanWidgets(),
      ),
      const Padding(padding: EdgeInsets.all(4)),
    ]);

    if (_currentSubscription != null) {
      widgets.add(
        ValidityWidget(
          currentSubscription: _currentSubscription,
          bonusData: _userDetails.bonusData,
        ),
      );
    }

    if (_currentSubscription!.productID == freeProductID) {
      if (widget.isOnboarding) {
        widgets.add(SkipSubscriptionWidget(freePlan: _freePlan));
      }
      widgets.add(
        SubFaqWidget(isOnboarding: widget.isOnboarding),
      );
    }

    // only active subscription can be renewed/canceled
    if (_hasActiveSubscription && _isStripeSubscriber) {
      widgets.add(_stripeRenewOrCancelButton());
    }

    if (_currentSubscription!.productID != freeProductID) {
      widgets.add(
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 40, 16, 4),
          child: MenuItemWidget(
            captionedTextWidget: CaptionedTextWidget(
              title: S.of(context).paymentDetails,
            ),
            menuItemColor: colorScheme.fillFaint,
            trailingWidget: Icon(
              Icons.chevron_right_outlined,
              color: colorScheme.strokeBase,
            ),
            singleBorderRadius: 4,
            alignCaptionedTextToLeft: true,
            onTap: () async {
              _redirectToPaymentPortal();
            },
          ),
        ),
      );
    }

    if (!widget.isOnboarding) {
      widgets.add(
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 0),
          child: MenuItemWidget(
            captionedTextWidget: CaptionedTextWidget(
              title: S.of(context).manageFamily,
            ),
            menuItemColor: colorScheme.fillFaint,
            trailingWidget: Icon(
              Icons.chevron_right_outlined,
              color: colorScheme.strokeBase,
            ),
            singleBorderRadius: 4,
            alignCaptionedTextToLeft: true,
            onTap: () async {
              // ignore: unawaited_futures
              _billingService.launchFamilyPortal(context, _userDetails);
            },
          ),
        ),
      );
      widgets.add(ViewAddOnButton(_userDetails.bonusData));
      widgets.add(const SizedBox(height: 80));
    }

    return SingleChildScrollView(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: widgets,
      ),
    );
  }

  // _redirectToPaymentPortal action allows the user to update
  // their stripe payment details
  void _redirectToPaymentPortal() async {
    final String paymentProvider = _currentSubscription!.paymentProvider;
    switch (_currentSubscription!.paymentProvider) {
      case stripe:
        await _launchStripePortal();
        break;
      case playStore:
        unawaited(
          launchUrlString(
            "https://play.google.com/store/account/subscriptions?sku=" +
                _currentSubscription!.productID +
                "&package=io.ente.photos",
          ),
        );
        break;
      case appStore:
        unawaited(launchUrlString("https://apps.apple.com/account/billing"));
        break;
      default:
        final String capitalizedWord = paymentProvider.isNotEmpty
            ? '${paymentProvider[0].toUpperCase()}${paymentProvider.substring(1).toLowerCase()}'
            : '';
        await showErrorDialog(
          context,
          S.of(context).sorry,
          S.of(context).contactToManageSubscription(capitalizedWord),
        );
    }
  }

  Future<void> _launchStripePortal() async {
    await _dialog.show();
    try {
      final String url = await _billingService.getStripeCustomerPortalUrl();
      await _dialog.hide();
      await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (BuildContext context) {
            return WebPage(S.of(context).paymentDetails, url);
          },
        ),
      ).then((value) => onWebPaymentGoBack);
    } catch (e) {
      await _dialog.hide();
      await showGenericErrorDialog(context: context, error: e);
    }
  }

  Widget _stripeRenewOrCancelButton() {
    final bool isRenewCancelled =
        _currentSubscription!.attributes?.isCancelled ?? false;
    if (isRenewCancelled && _userDetails.hasPaidAddon()) {
      return const SizedBox.shrink();
    }
    final String title = isRenewCancelled
        ? S.of(context).renewSubscription
        : S.of(context).cancelSubscription;
    return TextButton(
      child: Text(
        title,
        style: TextStyle(
          color: (isRenewCancelled
              ? colorScheme.primary700
              : colorScheme.textMuted),
        ),
      ),
      onPressed: () async {
        bool confirmAction = false;
        if (isRenewCancelled) {
          final choice = await showChoiceDialog(
            context,
            title: title,
            body: S.of(context).areYouSureYouWantToRenew,
            firstButtonLabel: S.of(context).yesRenew,
          );
          confirmAction = choice!.action == ButtonAction.first;
        } else {
          final choice = await showChoiceDialog(
            context,
            title: title,
            body: S.of(context).areYouSureYouWantToCancel,
            firstButtonLabel: S.of(context).yesCancel,
            secondButtonLabel: S.of(context).no,
            isCritical: true,
          );
          confirmAction = choice!.action == ButtonAction.first;
        }
        if (confirmAction) {
          await toggleStripeSubscription(isRenewCancelled);
        }
      },
    );
  }

  // toggleStripeSubscription, based on current auto renew status, will
  // toggle the auto renew status of the user's subscription
  Future<void> toggleStripeSubscription(bool isAutoRenewDisabled) async {
    await _dialog.show();
    try {
      isAutoRenewDisabled
          ? await _billingService.activateStripeSubscription()
          : await _billingService.cancelStripeSubscription();
      await _fetchSub();
    } catch (e) {
      showShortToast(
        context,
        isAutoRenewDisabled
            ? S.of(context).failedToRenew
            : S.of(context).failedToCancel,
      );
    }
    await _dialog.hide();
    if (!isAutoRenewDisabled && mounted) {
      await showTextInputDialog(
        context,
        title: S.of(context).askCancelReason,
        submitButtonLabel: S.of(context).send,
        hintText: S.of(context).optionalAsShortAsYouLike,
        alwaysShowSuccessState: true,
        textCapitalization: TextCapitalization.words,
        onSubmit: (String text) async {
          // indicates user cancelled the rename request
          if (text == "" || text.trim().isEmpty) {
            return;
          }
          try {
            await UserService.instance.sendFeedback(context, text);
          } catch (e, s) {
            logger.severe("Failed to send feedback", e, s);
          }
        },
      );
    }
  }

  List<Widget> _getStripePlanWidgets() {
    final List<Widget> planWidgets = [];
    bool foundActivePlan = false;
    for (final plan in _plans) {
      final productID = plan.stripeID;
      if (productID.isEmpty) {
        continue;
      }
      final isActive = _hasActiveSubscription &&
          _currentSubscription!.productID == productID;
      if (isActive) {
        foundActivePlan = true;
      }
      planWidgets.add(
        Material(
          child: InkWell(
            onTap: () async {
              if (isActive) {
                return;
              }
              // prompt user to cancel their active subscription form other
              // payment providers
              if (!_isStripeSubscriber &&
                  _hasActiveSubscription &&
                  _currentSubscription!.productID != freeProductID) {
                await showErrorDialog(
                  context,
                  S.of(context).sorry,
                  S.of(context).cancelOtherSubscription(
                        _currentSubscription!.paymentProvider,
                      ),
                );
                return;
              }
              final int addOnBonus =
                  _userDetails.bonusData?.totalAddOnBonus() ?? 0;
              if (_userDetails.getFamilyOrPersonalUsage() >
                  (plan.storage + addOnBonus)) {
                logger.warning(
                  " familyUsage ${convertBytesToReadableFormat(_userDetails.getFamilyOrPersonalUsage())}"
                  " plan storage ${convertBytesToReadableFormat(plan.storage)} "
                  "addOnBonus ${convertBytesToReadableFormat(addOnBonus)},"
                  "overshooting by ${convertBytesToReadableFormat(_userDetails.getFamilyOrPersonalUsage() - (plan.storage + addOnBonus))}",
                );
                await showErrorDialog(
                  context,
                  S.of(context).sorry,
                  S.of(context).youCannotDowngradeToThisPlan,
                );
                return;
              }
              String stripPurChaseAction = 'buy';
              if (_isStripeSubscriber && _hasActiveSubscription) {
                // confirm if user wants to change plan or not
                final result = await showChoiceDialog(
                  context,
                  title: S.of(context).confirmPlanChange,
                  body: S.of(context).areYouSureYouWantToChangeYourPlan,
                  firstButtonLabel: S.of(context).yes,
                );
                if (result!.action == ButtonAction.first) {
                  stripPurChaseAction = 'update';
                } else {
                  return;
                }
              }
              await Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (BuildContext context) {
                    return PaymentWebPage(
                      planId: plan.stripeID,
                      actionType: stripPurChaseAction,
                    );
                  },
                ),
              ).then((value) => onWebPaymentGoBack(value));
            },
            child: SubscriptionPlanWidget(
              storage: plan.storage,
              price: plan.price,
              period: plan.period,
              isActive: isActive && !_hideCurrentPlanSelection,
            ),
          ),
        ),
      );
    }
    if (!foundActivePlan && _hasActiveSubscription) {
      _addCurrentPlanWidget(planWidgets);
    }
    return planWidgets;
  }

  bool _isFreePlanUser() {
    return _currentSubscription != null &&
        freeProductID == _currentSubscription!.productID;
  }

  Widget _showSubscriptionToggle() {
    // return Container(
    //   padding: const EdgeInsets.fromLTRB(16, 32, 16, 6),
    //   child: Column(
    //     children: [
    //       RepaintBoundary(
    //         child: SizedBox(
    //           width: 250,
    //           child: Row(
    //             mainAxisSize: MainAxisSize.max,
    //             mainAxisAlignment: MainAxisAlignment.center,
    //             children: [
    //               Expanded(
    //                 child: SegmentedButton(
    //                   style: SegmentedButton.styleFrom(
    //                     selectedBackgroundColor:
    //                         getEnteColorScheme(context).fillMuted,
    //                     selectedForegroundColor:
    //                         getEnteColorScheme(context).textBase,
    //                     side: BorderSide(
    //                       color: getEnteColorScheme(context).strokeMuted,
    //                       width: 1,
    //                     ),
    //                   ),
    //                   segments: <ButtonSegment<bool>>[
    //                     ButtonSegment(
    //                       label: Text(S.of(context).monthly),
    //                       value: false,
    //                     ),
    //                     ButtonSegment(
    //                       label: Text(S.of(context).yearly),
    //                       value: true,
    //                     ),
    //                   ],
    //                   selected: {_showYearlyPlan},
    //                   onSelectionChanged: (p0) {
    //                     _showYearlyPlan = p0.first;
    //                     _filterStripeForUI();
    //                   },
    //                 ),
    //               ),
    //             ],
    //           ),
    //         ),
    //       ),
    //       const Padding(padding: EdgeInsets.all(8)),
    //     ],
    //   ),
    // );

    //

    return SubscriptionToggle();
  }

  void _addCurrentPlanWidget(List<Widget> planWidgets) {
    // don't add current plan if it's monthly plan but UI is showing yearly plans
    // and vice versa.
    if (_showYearlyPlan != _currentSubscription!.isYearlyPlan() &&
        _currentSubscription!.productID != freeProductID) {
      return;
    }
    int activePlanIndex = 0;
    for (; activePlanIndex < _plans.length; activePlanIndex++) {
      if (_plans[activePlanIndex].storage > _currentSubscription!.storage) {
        break;
      }
    }
    planWidgets.insert(
      activePlanIndex,
      Material(
        child: InkWell(
          onTap: () {},
          child: SubscriptionPlanWidget(
            storage: _currentSubscription!.storage,
            price: _currentSubscription!.price,
            period: _currentSubscription!.period,
            isActive: _currentSubscription!.isValid(),
          ),
        ),
      ),
    );
  }
}

class SubscriptionToggle extends StatefulWidget {
  const SubscriptionToggle({super.key});

  @override
  State<SubscriptionToggle> createState() => _SubscriptionToggleState();
}

class _SubscriptionToggleState extends State<SubscriptionToggle> {
  bool _isYearly = false;
  @override
  Widget build(BuildContext context) {
    const borderPadding = 2.5;
    const spaceBetweenButtons = 4.0;
    final textTheme = getEnteTextTheme(context);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 32),
      child: LayoutBuilder(
        builder: (context, constrains) {
          final widthOfButton = (constrains.maxWidth -
                  (borderPadding * 2) -
                  spaceBetweenButtons) /
              2;
          return Container(
            decoration: BoxDecoration(
              color: const Color.fromRGBO(242, 242, 242, 1),
              borderRadius: BorderRadius.circular(50),
            ),
            padding: const EdgeInsets.symmetric(
              vertical: borderPadding,
              horizontal: borderPadding,
            ),
            width: double.infinity,
            child: Stack(
              children: [
                Row(
                  children: [
                    GestureDetector(
                      onTap: () {
                        setState(() {
                          _isYearly = false;
                        });
                      },
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          vertical: 8,
                        ),
                        width: widthOfButton,
                        child: Center(
                          child: Text(
                            "Monthly",
                            style: textTheme.bodyFaint,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: spaceBetweenButtons),
                    GestureDetector(
                      onTap: () {
                        setState(() {
                          _isYearly = true;
                        });
                      },
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          vertical: 8,
                        ),
                        width: widthOfButton,
                        child: Center(
                          child: Text(
                            "Yearly",
                            style: textTheme.bodyFaint,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
                AnimatedPositioned(
                  duration: const Duration(milliseconds: 500),
                  curve: Curves.easeInOutQuart,
                  left: _isYearly ? widthOfButton + spaceBetweenButtons : 0,
                  child: Container(
                    width: widthOfButton,
                    height: 40,
                    decoration: BoxDecoration(
                      color: const Color.fromRGBO(255, 255, 255, 1),
                      borderRadius: BorderRadius.circular(50),
                    ),
                    child: AnimatedSwitcher(
                      duration: const Duration(milliseconds: 500),
                      switchInCurve: Curves.easeInOutExpo,
                      switchOutCurve: Curves.easeInOutExpo,
                      child: Text(
                        key: ValueKey(_isYearly),
                        _isYearly ? "Yearly" : "Monthly",
                        style: textTheme.body,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}
